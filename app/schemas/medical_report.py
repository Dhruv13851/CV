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
            "Green=inside the applicable range, Yellow=borderline, "
            "Red=outside. Null when no range applies or the result "
            "is not numerically comparable."
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
        description="Doctor name, if stated."
    )

    report_title: Optional[str] = Field(
        None,
        description="Report title exactly as written; null if not printed."
    )

    sections: List[TestSection] = Field(
        ...,
        description="All test sections in the report."
    )

    @model_validator(mode="after")
    def drop_empty_sections(self):
        """Discard categories that hold no tests of their own.

        A printed header whose rows all belong to its sub-headers (URINE
        ROUTINE on sample2.pdf page 8) owns nothing, and the prompt says to
        leave it out. The model complied in 4 runs out of 5. An empty section
        carries no data, so dropping it here makes the miss unrepresentable
        instead of merely unlikely.
        """
        self.sections = [s for s in self.sections if s.tests]
        return self