"""Dictionary ingestion must normalize whatever shape the business supplies."""

import json

import pytest

from ml_engine.dictionary import DictionaryParseError, parse_dictionary


def test_csv_with_arbitrary_headers_is_normalized():
    payload = (
        b"Field Name;Business Definition;Source System;Available at prediction time;Owner\n"
        b"customer_id;Unique customer key;CRM;yes;data-team\n"
        b"closed_reason;Recorded when the account closes;Billing;no;billing-team\n"
    )
    dictionary = parse_dictionary(payload, source_format="csv")
    first = dictionary.for_column("customer_id")
    assert first.business_definition == "Unique customer key"
    assert first.source_system == "CRM"
    assert first.available_at_prediction_time is True
    assert dictionary.for_column("closed_reason").available_at_prediction_time is False


def test_json_list_form_is_supported():
    payload = json.dumps(
        [{"column": "age", "description": "Customer age", "known_at_prediction_time": "true"}]
    ).encode()
    dictionary = parse_dictionary(payload, source_format="json")
    assert dictionary.columns[0].column == "age"
    assert dictionary.columns[0].available_at_prediction_time is True


def test_json_object_form_is_supported():
    payload = json.dumps({"age": {"definition": "Customer age"}, "city": "Billing city"}).encode()
    dictionary = parse_dictionary(payload, source_format="json")
    assert {c.column for c in dictionary.columns} == {"age", "city"}
    assert dictionary.for_column("city").business_definition == "Billing city"


def test_unknown_source_columns_are_preserved_as_extra():
    payload = b"column,definition,jira_ticket\nage,Customer age,DATA-42\n"
    dictionary = parse_dictionary(payload, source_format="csv")
    assert dictionary.columns[0].extra == {"jira_ticket": "DATA-42"}


def test_reconciliation_reports_both_directions():
    payload = b"column,definition\nage,Customer age\nghost,Not in the dataset\n"
    dictionary = parse_dictionary(payload, source_format="csv", dataset_columns=["age", "city"])
    assert dictionary.unmatched_columns == ["ghost"]
    assert dictionary.undocumented_columns == ["city"]


def test_xlsx_is_supported():
    openpyxl = pytest.importorskip("openpyxl")
    import io

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Column", "Definition", "Available at prediction time"])
    sheet.append(["age", "Customer age", "yes"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    dictionary = parse_dictionary(buffer.getvalue(), source_format="xlsx")
    assert dictionary.for_column("age").available_at_prediction_time is True


def test_unsupported_format_is_rejected():
    with pytest.raises(DictionaryParseError, match="Unsupported"):
        parse_dictionary(b"x", source_format="pdf")


def test_file_without_a_column_name_is_rejected():
    with pytest.raises(DictionaryParseError, match="No column name"):
        parse_dictionary(b"foo,bar\n1,2\n", source_format="csv")


def test_invalid_json_is_rejected_clearly():
    with pytest.raises(DictionaryParseError, match="not valid JSON"):
        parse_dictionary(b"{oops", source_format="json")
