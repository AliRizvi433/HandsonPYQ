
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document


class DocumentLoader:

    def __init__(
        self,
        policies_path="data/policies.json",
        hris_path="data/hris_api.json"
    ):
        self.policies_path = Path(policies_path)
        self.hris_path = Path(hris_path)

    def create_documents(self) -> List[Document]:

        documents = []

        # Load JSON files
        with open(self.policies_path, "r", encoding="utf-8") as f:
            policies = json.load(f)

        with open(self.hris_path, "r", encoding="utf-8") as f:
            employees = json.load(f)

        # Convert policies to Documents
        for policy in policies:

            content = (
                f"Policy ID: {policy['id']}\n"
                f"Policy Title: {policy['title']}\n"
                f"Category: {policy['category']}\n"
                f"Effective Year: {policy['effective_year']}\n"
                f"Description: {policy['description']}\n"
                f"Active: {policy['active']}"
            )

            metadata = {
                "policy_id": policy["id"],
                "title": policy["title"],
                "category": policy["category"],
                "effective_year": policy["effective_year"],
                "active": policy["active"],
                "type": "policy"
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

        # Convert employees to Documents
        for employee in employees:

            pto_details = employee.get("pto_details", {})

            pto_formatted = ", ".join(
                f"{key}: {value}"
                for key, value in pto_details.items()
            )

            content = (
                f"Employee Name: {employee['name']}\n"
                f"Employee ID: {employee['employee_id']}\n"
                f"Employment Status: {employee['employment_status']}\n"
                f"Location: {employee['location']}\n"
                f"PTO Balance: {employee['pto_balance']} days\n"
                f"PTO Details: {pto_formatted}"
            )

            metadata = {
                "name": employee["name"],
                "employee_id": employee["employee_id"],
                "employment_status": employee["employment_status"],
                "location": employee["location"],
                "pto_balance": employee["pto_balance"],
                "type": "employee"
            }

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata
                )
            )

        return documents
