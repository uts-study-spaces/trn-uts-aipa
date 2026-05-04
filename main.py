"""Command-line entry point for a quick ticket prediction."""

from customer_support_ai.predict import predict_ticket


def main() -> None:
    ticket = input("Paste a support ticket: ").strip()
    if not ticket:
        print("No ticket text provided.")
        return

    result = predict_ticket(ticket)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
