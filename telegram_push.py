import os
from pathlib import Path

import requests


OUTPUT_DIR = Path("output")

TELEGRAM_LIMIT = 4000


def send_telegram(message):

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN secret is missing."
        )

    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID secret is missing."
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

        response.raise_for_status()

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
