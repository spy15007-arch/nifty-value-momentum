import os
from pathlib import Path

import requests


OUTPUT_DIR = Path("output")

TELEGRAM_LIMIT = 4000


def normalize_secret(value):
    return (value or "").strip().strip('"').strip("'")


def send_telegram(message):

    token = normalize_secret(
        os.getenv("TELEGRAM_BOT_TOKEN")
    )

    chat_id = normalize_secret(
        os.getenv("TELEGRAM_CHAT_ID")
    )

    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN secret is missing."
        )

    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID secret is missing."
        )

    if chat_id.startswith("https://t.me/") or chat_id.startswith("t.me/"):
        raise RuntimeError(
            "TELEGRAM_CHAT_ID must be a numeric chat ID or @username, not a Telegram URL."
        )

    url = (
        f"https://api.telegram.org/bot"
        f"{token}/sendMessage"
    )

    chunks = [
        message[i:i + TELEGRAM_LIMIT]
        for i in range(
            0,
            len(message),
            TELEGRAM_LIMIT,
        )
    ]

    for chunk in chunks:

        response = requests.post(
            url,
            data={
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            },
            timeout=30,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            error_text = response.text[:500]
            raise RuntimeError(
                f"Telegram API rejected chat_id={chat_id!r}: {error_text}"
            ) from exc

        result = response.json()

        if not result.get("ok"):

            raise RuntimeError(
                f"Telegram error: {result}"
            )


def main():

    message_file = (
        OUTPUT_DIR
        / "telegram_message.txt"
    )

    if not message_file.exists():

        raise RuntimeError(
            "telegram_message.txt was not created."
        )

    message = (
        message_file
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    if not message:

        raise RuntimeError(
            "Telegram message is empty."
        )

    send_telegram(
        message
    )

    print(
        "Telegram message sent successfully."
    )


if __name__ == "__main__":

    main()
