#!/usr/bin/env python3
"""
Fixed M-PESA SMS Data Preparation Script
This script fixes the date/time extraction issues in the original data preparation
"""

import re
import json
import pandas as pd
from typing import Dict, Any, Optional

def parse_mpesa_sms_fixed(sms: str, date_time: str = None) -> Dict[str, Any]:
    """
    Enhanced M-PESA SMS parser with improved date/time extraction
    """
    # Extract transaction ID (first word)
    transaction_id = sms.split()[0] if sms else None

    # Extract amount - improved pattern to handle various formats
    amount_match = re.search(r"Ksh\s*([0-9,]+(?:\.[0-9]{2})?)", sms)
    amount = amount_match.group(1).replace(",", "") if amount_match else None

    # Transaction type detection (same as before but organized)
    transaction_type = "unknown"
    transaction_patterns = [
        (r"received\s+from", "received"),
        (r"have\s+received", "received"),
        (r"sent\s+to", "sent"),
        (r"withdrew?\s+at", "withdrawn"),
        (r"failed", "failed"),
        (r"buy\s+goods", "buy_goods"),
        (r"paid\s+to", "paybill"),
        (r"bought.*?airtime", "airtime"),
        (r"airtime\s+purchase", "airtime"),
        (r"deposited\s+to", "deposit"),
        (r"reversed", "reversal"),
        (r"bill\s+payment", "bill_payment"),
        (r"loan\s+disbursed", "loan"),
        (r"fuliza", "fuliza"),
    ]

    for pattern, ttype in transaction_patterns:
        if re.search(pattern, sms, re.IGNORECASE):
            transaction_type = ttype
            break

    # Enhanced counterparty extraction
    counterparty = None

    # Try different patterns based on transaction type
    counterparty_patterns = [
        r"(?:from|to)\s+(CUSTOMER_\w+)",
        r"(?:from|to)\s+([A-Z][A-Za-z0-9_\s]+?)\s+(?:\d+\s+)?on",
        r"(?:from|to)\s+([A-Z][A-Za-z0-9_\s]+?)\s+\d{2}/",
        r"(CUSTOMER_\w+)",
        r"(AGENT_\w+)",
    ]

    for pattern in counterparty_patterns:
        match = re.search(pattern, sms)
        if match:
            counterparty = match.group(1).strip()
            break

    # Special handling for airtime
    if transaction_type == "airtime":
        counterparty = "self"

    # FIXED DATE/TIME EXTRACTION
    parsed_date_time = None

    if date_time:
        parsed_date_time = date_time
    else:
        # Multiple patterns to catch different date formats
        date_patterns = [
            # Pattern 1: d/m/yy at h:mm AM/PM (single digits allowed)
            r"(\d{1,2}/\d{1,2}/\d{2}\s+at\s+\d{1,2}:\d{2}\s+(?:AM|PM))",
            # Pattern 2: on d/m/yy at h:mm AM/PM
            r"on\s+(\d{1,2}/\d{1,2}/\d{2}\s+at\s+\d{1,2}:\d{2}\s+(?:AM|PM))",
            # Pattern 3: Just the date and time part
            r"(\d{1,2}/\d{1,2}/\d{2}.*?\d{1,2}:\d{2}\s+(?:AM|PM))",
        ]

        for pattern in date_patterns:
            date_match = re.search(pattern, sms)
            if date_match:
                parsed_date_time = date_match.group(1).strip()
                break

    # Extract balance
    balance_match = re.search(r"balance.*?Ksh\s*([0-9,]+(?:\.[0-9]{2})?)", sms, re.IGNORECASE)
    balance = balance_match.group(1).replace(",", "") if balance_match else None

    return {
        "transaction_id": transaction_id,
        "amount": amount,
        "transaction_type": transaction_type,
        "counterparty": counterparty,
        "date_time": parsed_date_time,
        "balance": balance
    }

def test_parser():
    """Test the fixed parser with sample SMS messages"""
    test_messages = [
        "TAG2DHJIM8 Confirmed. Ksh460.00 sent to CUSTOMER_6D8DC478 16/1/25 at 11:18 AM. New M-PESA balance is Ksh5,612.59.",
        "TAH1K18FRF confirmed.You bought Ksh200.00 of airtime on 17/1/25 at 7:54 PM.New M-PESA balance is Ksh5,412.59.",
        "QFC3D45G7 Confirmed. You have received Ksh500 from Customer_1 XXXXXXX on 12/08/24 at 11:32 AM. New M-PESA balance is Ksh1,250.",
        "TAI7MMSHXP Confirmed. On 18/1/25 at 2:13 PM Give Ksh25,600.00 cash to CUSTOMER_F2E00682-PESA balance is Ksh28,987.59."
    ]

    print("🧪 Testing fixed parser...")
    for i, sms in enumerate(test_messages, 1):
        result = parse_mpesa_sms_fixed(sms)
        print(f"\nTest {i}:")
        print(f"SMS: {sms}")
        print(f"Parsed: {json.dumps(result, indent=2)}")

        # Check if date_time was extracted
        if result['date_time']:
            print("✅ Date/Time extracted successfully!")
        else:
            print("❌ Date/Time extraction failed")

def regenerate_training_data():
    """Regenerate the training data with fixed date/time extraction"""

    print("🔄 Regenerating training data with fixed date/time extraction...")

    # Read the current training data to get the SMS messages
    input_file = "output/mpesa_basic.jsonl"
    output_file = "output/mpesa_basic_fixed.jsonl"

    fixed_data = []

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                sms_text = data['input']

                # Re-parse with fixed function
                parsed = parse_mpesa_sms_fixed(sms_text)

                # Create new training example
                new_example = {
                    "input": sms_text,
                    "output": json.dumps(parsed)
                }

                fixed_data.append(new_example)

                if line_num <= 5:  # Show first 5 examples
                    print(f"\nExample {line_num}:")
                    print(f"Input: {sms_text}")
                    print(f"Old output: {data['output']}")
                    print(f"New output: {new_example['output']}")

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue

    # Save fixed data
    with open(output_file, 'w') as f:
        for example in fixed_data:
            f.write(json.dumps(example) + '\n')

    print(f"\n✅ Fixed training data saved to: {output_file}")
    print(f"📊 Processed {len(fixed_data)} examples")

    # Verify date extraction success rate
    with_dates = sum(1 for ex in fixed_data if '"date_time": null' not in ex['output'])
    date_success_rate = (with_dates / len(fixed_data)) * 100
    print(f"📅 Date extraction success rate: {date_success_rate:.1f}% ({with_dates}/{len(fixed_data)})")

    return output_file

if __name__ == "__main__":
    # First test the parser
    test_parser()

    print("\n" + "="*60)

    # Then regenerate training data
    fixed_file = regenerate_training_data()

    print(f"\n🎉 Data preparation fixed!")
    print(f"📁 Use this file for training: {fixed_file}")
