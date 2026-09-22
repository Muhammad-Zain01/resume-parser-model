"""Pydantic contract for the JSON produced by the resume parser model.

The model should return one :class:`ResumeOutput` object per resume. Missing
single values are ``None`` and missing repeated sections are empty lists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResumeModel(BaseModel):
    """Base configuration shared by every output object."""

    model_config = ConfigDict(extra="forbid")


class PersonalInformation(ResumeModel):
    full_name: str | None = None
    professional_title: str | None = None
    email: str | None = None
    phone_numbers: list[str] = Field(default_factory=list)
    address: str | None = None
    city: str | None = None
    state_or_region: str | None = None
    postal_code: str | None = None
    country: str | None = None
    linkedin: str | None = None
    portfolio: str | None = None
    other_links: list[str] = Field(default_factory=list)


class SpokenLanguage(ResumeModel):
    language: str | None = None
    proficiency: str | None = None


class Skills(ResumeModel):
    technical: list[str] = Field(default_factory=list)
    tools_and_software: list[str] = Field(default_factory=list)
    domain: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)
    spoken_languages: list[SpokenLanguage] = Field(default_factory=list)


class WorkExperience(ResumeModel):
    job_title: str | None = None
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool | None = None
    description: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)


class Education(ResumeModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    graduation_date: str | None = None
    gpa: str | None = None
    honors: list[str] = Field(default_factory=list)


class Certification(ResumeModel):
    name: str | None = None
    type: str | None = None
    issuer: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    credential_id: str | None = None
    description: str | None = None


class Project(ResumeModel):
    name: str | None = None
    role: str | None = None
    description: str | None = None
    technologies: list[str] = Field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    url: str | None = None


class AwardOrHonor(ResumeModel):
    name: str | None = None
    issuer: str | None = None
    date: str | None = None
    description: str | None = None


class Publication(ResumeModel):
    title: str | None = None
    publisher: str | None = None
    date: str | None = None
    url: str | None = None


class VolunteerExperience(ResumeModel):
    organization: str | None = None
    role: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str | None = None


class ProfessionalMembership(ResumeModel):
    organization: str | None = None
    role: str | None = None
    date: str | None = None


class Reference(ResumeModel):
    name: str | None = None
    relationship: str | None = None
    company: str | None = None
    contact: str | None = None


class AdditionalSection(ResumeModel):
    section_name: str | None = None
    content: str | None = None


class ResumeOutput(ResumeModel):
    """Canonical structured output generated from one resume."""

    personal_information: PersonalInformation = Field(default_factory=PersonalInformation)
    professional_summary: str | None = None
    career_objective: str | None = None
    target_roles: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: list[Certification] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    awards_and_honors: list[AwardOrHonor] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)
    volunteer_experience: list[VolunteerExperience] = Field(default_factory=list)
    professional_memberships: list[ProfessionalMembership] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    additional_sections: list[AdditionalSection] = Field(default_factory=list)
