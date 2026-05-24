"""
main.py — FastAPI Voice Translation Proxy Server
=================================================
Data flow per request:
  C++ Client (WAV audio via multipart/form-data)
      ↓
  /translate endpoint
      ↓
  [STT]  SpeechRecognition → Google Free Web API → English text
      ↓
  [TTS]  DeepL API → Translated text (target language)
      ↓
  [TTS]  gTTS → MP3 audio buffer (in-memory, no disk writes)
      ↓
  Response (audio/mpeg, known Content-Length) back to C++ client
"""

import io
import os
import logging
import tempfile
from urllib.parse import quote

import deepl
import speech_recognition as sr
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from gtts import gTTS

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

# Load DEEPL_AUTH_KEY (and any other secrets) from a local .env file.
# On Render, set these as environment variables in the dashboard instead.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice-proxy")

app = FastAPI(title="Voice Translation Proxy", version="1.0.0")

# ---------------------------------------------------------------------------
# DeepL client (initialised once at startup, reused across requests)
# ---------------------------------------------------------------------------

_DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")
if not _DEEPL_AUTH_KEY:
    raise RuntimeError(
        "DEEPL_AUTH_KEY is not set. "
        "Add it to your .env file or Render environment variables."
    )

deepl_client = deepl.Translator(_DEEPL_AUTH_KEY)

# ---------------------------------------------------------------------------
# Language-code mapping: DeepL codes → gTTS BCP-47 codes
#
# DeepL uses uppercase ISO codes (sometimes with region, e.g. "PT-BR").
# gTTS uses lowercase BCP-47 tags.  A few special cases:
#   • Hebrew: DeepL="HE", gTTS="iw"  (old ISO 639-1, still used by Google TTS)
#   • Chinese Simplified: DeepL="ZH" / "ZH-HANS", gTTS="zh-CN"
#   • Chinese Traditional: DeepL="ZH-HANT", gTTS="zh-TW"
#   • Portuguese Brazil: DeepL="PT-BR", gTTS="pt"  (gTTS defaults to BR accent)
#   • Portuguese Portugal: DeepL="PT-PT", gTTS="pt"
# ---------------------------------------------------------------------------

DEEPL_TO_GTTS: dict[str, str] = {
    "BG": "bg",
    "CS": "cs",
    "DA": "da",
    "DE": "de",
    "EL": "el",
    "EN": "en",
    "EN-GB": "en",
    "EN-US": "en",
    "ES": "es",
    "ET": "et",
    "FI": "fi",
    "FR": "fr",
    "HU": "hu",
    "ID": "id",
    "IT": "it",
    "JA": "ja",
    "KO": "ko",
    "LT": "lt",
    "LV": "lv",
    "NB": "no",   # Norwegian Bokmål
    "NL": "nl",
    "PL": "pl",
    "PT": "pt",
    "PT-BR": "pt",
    "PT-PT": "pt",
    "RO": "ro",
    "RU": "ru",
    "SK": "sk",
    "SL": "sl",
    "SV": "sv",
    "TR": "tr",
    "UK": "uk",
    "ZH": "zh-CN",
    "ZH-HANS": "zh-CN",
    "ZH-HANT": "zh-TW",
    # Special case — gTTS still uses the legacy "iw" tag for Hebrew
    "HE": "iw",
}


def deepl_code_to_gtts(deepl_lang: str) -> str:
    """
    Convert a DeepL target language code to the gTTS language tag.

    Raises ValueError if the code is unrecognised so the caller can
    return a clean 400 error to the client.
    """
    key = deepl_lang.upper()
    if key not in DEEPL_TO_GTTS:
        raise ValueError(
            f"Unsupported target language '{deepl_lang}'. "
            f"Supported codes: {sorted(DEEPL_TO_GTTS)}"
        )
    return DEEPL_TO_GTTS[key]


# ---------------------------------------------------------------------------
# Step 1 — Speech-to-Text helper
# ---------------------------------------------------------------------------

def transcribe_audio(wav_bytes: bytes) -> str:
    """
    Transcribe raw WAV bytes to English text via Google's free web STT API.

    SpeechRecognition cannot read from a BytesIO object directly, so we
    write to a NamedTemporaryFile as a shim and clean it up manually.

    WHY delete=False + manual cleanup:
        On Windows, NamedTemporaryFile(delete=True) keeps an exclusive open
        handle on the file for the lifetime of the 'with' block. When
        sr.AudioFile then calls wave.open() on the same path, Windows refuses
        to grant a second handle → PermissionError [Errno 13].
        The fix is to close (and therefore release) the handle ourselves
        before passing the path to SpeechRecognition, then delete manually
        in a 'finally' block so no temp files are ever left behind.

    Returns:
        Transcribed English string.

    Raises:
        HTTPException 400 — audio was unintelligible.
        HTTPException 502 — Google STT service unreachable.
    """
    recogniser = sr.Recognizer()
    tmp_path = None

    try:
        # delete=False, so we can close the write handle before SR opens it.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(wav_bytes)
            # tmp is fully closed (and its handle released) when this 'with' exits.

        # Now the OS handle is free — SpeechRecognition can open the file normally.
        with sr.AudioFile(tmp_path) as source:
            audio_data = recogniser.record(source)

    finally:
        # Always remove the temp file, even if SR raised an exception.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    try:
        text = recogniser.recognize_google(audio_data, language="en-US")
        logger.info("STT result: %r", text)
        return text

    except sr.UnknownValueError:
        # Google could not understand the audio
        raise HTTPException(
            status_code=400,
            detail="Speech recognition could not understand the audio. "
                   "Please speak clearly and try again.",
        )
    except sr.RequestError as exc:
        # Network or API error reaching Google
        raise HTTPException(
            status_code=502,
            detail=f"Google Speech Recognition service unavailable: {exc}",
        )


# ---------------------------------------------------------------------------
# Step 2 — Translation helper
# ---------------------------------------------------------------------------

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate *text* to *target_lang* using the DeepL API.

    DeepL accepts uppercase language codes such as "JA", "FR", "PT-BR".

    Returns:
        Translated string.

    Raises:
        HTTPException 400 — language code unrecognised by DeepL.
        HTTPException 502 — DeepL API error.
    """
    try:
        result = deepl_client.translate_text(
            text,
            source_lang="EN",
            target_lang=target_lang.upper(),
        )
        translated = result.text  # type: ignore[union-attr]
        logger.info("DeepL [%s]: %r → %r", target_lang, text, translated)
        return translated

    except deepl.exceptions.AuthorizationException as exc:
        raise HTTPException(status_code=500, detail=f"DeepL auth error: {exc}")
    except deepl.exceptions.DeepLException as exc:
        # Covers quota exceeded, bad language code, network errors, etc.
        raise HTTPException(status_code=502, detail=f"DeepL API error: {exc}")


# ---------------------------------------------------------------------------
# Step 3 — Text-to-Speech helper
# ---------------------------------------------------------------------------

def synthesise_speech(text: str, gtts_lang: str) -> io.BytesIO:
    """
    Convert *text* to MP3 audio using gTTS (Google Text-to-Speech).

    Everything is kept in memory — no files are written to disk.

    Returns:
        BytesIO buffer containing the MP3 payload, seeked to position 0.
    """
    tts = gTTS(text=text, lang=gtts_lang, slow=False)

    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)   # write MP3 bytes directly into memory
    mp3_buffer.seek(0)            # rewind so the caller can read from the start

    logger.info("TTS synthesised %d bytes of MP3 (lang=%s)", mp3_buffer.getbuffer().nbytes, gtts_lang)
    return mp3_buffer


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Watchdog / cold-start check")
def health_check() -> JSONResponse:
    """
    Lightweight endpoint polled by the C++ client to detect when the
    Render instance has finished its cold-start.

    Always returns HTTP 200 immediately.
    """
    return JSONResponse({"status": "ok", "message": "Server online"})


@app.post("/translate", summary="STT → Translate → TTS pipeline")
def translate(
    audio: UploadFile = File(..., description="WAV audio file captured by the C++ client"),
    target_lang: str = Form(..., description="DeepL target language code, e.g. JA, FR, ES"),
) -> Response:
    """
    Full pipeline: WAV audio → English text → translated text → MP3 response.

    WHY plain 'def' and not 'async def':
        All work here (SpeechRecognition, DeepL, gTTS) is synchronous/blocking.
        Marking the function 'async' without awaiting anything inside it would
        run those blocking calls on the event loop thread and stall every other
        request. FastAPI automatically runs plain 'def' endpoints in a thread
        pool executor, so concurrency is handled correctly without any awaits.

    Multipart form fields expected from the C++ client (libcurl):
      • audio      — the WAV file
      • target_lang — e.g. "JA"

    Returns an audio/mpeg stream the client can play immediately.
    """
    logger.info("Received /translate request: target_lang=%r, filename=%r", target_lang, audio.filename)

    # ------------------------------------------------------------------
    # 0. Validate the target language code before doing any heavy work
    # ------------------------------------------------------------------
    try:
        gtts_lang = deepl_code_to_gtts(target_lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 1. Read the uploaded WAV bytes from the request body
    #    audio.file is a SpooledTemporaryFile — .read() is synchronous.
    # ------------------------------------------------------------------
    wav_bytes = audio.file.read()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # ------------------------------------------------------------------
    # 2. STT — transcribe to English text
    # ------------------------------------------------------------------
    english_text = transcribe_audio(wav_bytes)

    # ------------------------------------------------------------------
    # 3. Translate — English → target language
    # ------------------------------------------------------------------
    translated_text = translate_text(english_text, target_lang)

    # ------------------------------------------------------------------
    # 4. TTS — translated text → MP3 buffer (in memory)
    # ------------------------------------------------------------------
    mp3_buffer = synthesise_speech(translated_text, gtts_lang)

    # ------------------------------------------------------------------
    # 5. Return the MP3 to the C++ client as a complete Response.
    #
    # WHY Response instead of StreamingResponse:
    #   StreamingResponse uses chunked transfer-encoding, which omits
    #   Content-Length. The C++ client then has no way to know the total
    #   size upfront and must realloc its buffer repeatedly as chunks
    #   arrive. Response reads the BytesIO into memory once, sets an exact
    #   Content-Length header, and lets the client pre-allocate correctly.
    #   Since the MP3 is already fully built in memory (BytesIO), there is
    #   no benefit to streaming it anyway.
    #
    # WHY urllib.parse.quote for headers:
    #   HTTP headers are ASCII-only. The old latin-1 encode+replace approach
    #   silently turned Japanese/Hebrew/etc. characters into '???????' —
    #   useless to the client. quote() percent-encodes every non-ASCII byte
    #   (e.g. "שלום" → "%D7%A9%D7%9C%D7%95%D7%9D"), which is 100%
    #   ASCII-safe, lossless, and trivially decoded on the C++ side with
    #   a single url-decode call (curl_easy_unescape or similar).
    # ------------------------------------------------------------------
    mp3_bytes = mp3_buffer.read()   # materialise the buffer for Content-Length

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={
            "X-Translated-Text": quote(translated_text),
            "X-Source-Text":     quote(english_text),
        },
    )
