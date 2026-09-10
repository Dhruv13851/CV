from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field, model_validator


class Patient(BaseModel):

    name: Optional[str] = Field(
        None,
        description="Patient name exactly as written in the report; null if not printed."
    )

    age: Optional[int] = Field(
        None,
        description="Patient age in years, if stated."
    )

    gender: Optional[str] = Field(
        None,
        description="Patient gender/sex, if stated."
    )


class ReferenceRange(BaseModel):

    label: Optional[str] = Field(
        None,
        description="Reference category, if stated(Return only lable)."
    )

    min_val: Optional[float] = Field(
        None,
        description="Lower limit; null if absent."
    )

    max_val: Optional[float] = Field(
        None,
        description="Upper limit; null if absent."
    )


class TestResult(BaseModel):

    name: str = Field(
        ...,
        description="Test name exactly as written."
    )

    result: Union[float, str] = Field(
        ...,
        description="Reported result; string for qualitative or range values."
    )

    unit: Optional[str] = Field(
        None,
        description="Reported unit, if present."
    )

    reference_ranges: List[ReferenceRange] = Field(
        default_factory=list,
        description="All reference ranges shown for the test."
    )

    indicator: Optional[Literal["Green", "Yellow", "Red"]] = Field(
        None,
        description=(
            "Green=healthy, Yellow=one band beyond healthy (borderline), "
            "Red=further than that. With a single range, Green=inside "
            "(boundaries included) and Red=outside. Null when no range "
            "applies, the result is not numeric, or it falls in no band."
        )
    )


class TestSection(BaseModel):

    category_name: str = Field(
        ...,
        description="Test section/category name."
    )

    tests: List[TestResult] = Field(
        ...,
        description="All tests in this section."
    )


class MedicalReport(BaseModel):

    patient: Patient = Field(
        ...,
        description="Patient information from the report."
    )

    lab_name: Optional[str] = Field(
        None,
        description="Laboratory/facility name, if stated."
    )

    doctor_name: Optional[str] = Field(
        None,
        description=(
            "Name of the doctor who signed/reported, as printed at the foot "
            "of the page; null if not printed."
        )
    )

    referring_doctor: Optional[str] = Field(
        None,
        description=(
            "The 'Referred by' / 'Ref. By' value from the patient header, "
            "exactly as printed. Often a hospital rather than a person."
        )
    )

    report_date: Optional[str] = Field(
        None,
        description=(
            "Report/registration date exactly as printed, including time if "
            "shown. Never reformatted."
        )
    )

    report_title: Optional[str] = Field(
        None,
        description="Report title exactly as written; null if not printed."
    )

    page_patients: List[str] = Field(
        default_factory=list,
        description=(
            "The patient name printed in the header of EVERY page, in page "
            "order, one entry per page - repeat the same name when every page "
            "shows it. A transcription of each header, never a judgement "
            "about whether they agree. Empty string for a page with no name."
        )
    )

    sections: List[TestSection] = Field(
        ...,
        description="All test sections in the report."
    )

    @model_validator(mode="after")
    def normalise_sections(self):
        """One section per printed category, and never an empty one.

        Two defects the model repeats on sample2.pdf, both of which the prompt
        already forbids and neither of which a consumer can undo:

        A category printed once but written twice (CRP came back as a second
        "Biochemistry" section in 3 runs of 3, extraction.md:16). Anyone
        keying results by category silently loses the first run to the
        second, so the later runs are appended to the first in page order.

        A header whose rows all belong to its sub-headers (URINE ROUTINE on
        page 8, extraction.md:17) owns nothing. It complied in 4 runs of 5.

        Doing it here makes both misses unrepresentable rather than merely
        unlikely. Note this is normalisation, never rejection - a validator
        that raised would put the whole report inside pydantic's error
        message, and from there into the SSE frame and the logs.
        """
        merged: dict[str, TestSection] = {}

        for section in self.sections:
            # Matched on the trimmed name, kept under the printed one. The
            # prompt no longer asks the model to avoid duplicates at all, so
            # "Biochemistry " must not slip past as a second section.
            key = section.category_name.strip()
            first = merged.get(key)
            if first is None:
                merged[key] = section
            else:
                first.tests.extend(section.tests)

        self.sections = [s for s in merged.values() if s.tests]
        return self


def reject_unusable(report) -> None:
    """Raise a clean ValueError when a validated report must not be returned.

    Both cases mean "the model answered, but this is not one patient's
    report". Deliberately NOT a pydantic validator: one that raises puts the
    whole report - names, values - inside the error message, and from there
    into the SSE frame and the logs.

    Mixed uploads used to rely on extraction.md alone, which asks the model to
    notice the mismatch AND suppress every section. Measured on two pages from
    two patients, that held in 1 run of 3; one of the failures returned both
    patients' 45 tests under a single name. page_patients replaces that
    judgement with transcription the code can compare.
    """
    reject_mixed_patients(report.page_patients)

    if not report.sections:
        raise ValueError(
            "No test results could be read from this upload. "
            "The pages may show more than one patient, or may "
            "not be a medical report."
        )


def reject_mixed_patients(page_patients) -> None:
    """Raise if the pages carry more than one patient name.

    Split out of reject_unusable so main.py can run it on the HEADER event.
    page_patients is written before `sections`, so every name is known ~4s in
    - long before the sections are streamed. Checking only at the end meant a
    consumer rendering the provisional section events saw one patient's
    results under another's header for nine seconds before the error frame.
    """
    if not isinstance(page_patients, list):
        return

    seen = {
        "".join(name.split()).lower()
        for name in page_patients
        if isinstance(name, str) and name.strip()
    }

    # the count, never the names - this string reaches the client and the logs
    if len(seen) > 1:
        raise ValueError(
            f"These pages belong to more than one patient: {len(seen)} "
            "different names are printed across them. Upload one patient's "
            "report at a time."
        )
