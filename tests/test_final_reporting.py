from __future__ import annotations

import pytest

from arxiv_topic_classifier.config import ProjectPaths
from arxiv_topic_classifier.final_reporting import FinalReportValidationError, run_final_report
from arxiv_topic_classifier.web import create_app


def test_final_report_refuses_stale_six_class_artifacts(
    trained_project: ProjectPaths,
) -> None:
    with pytest.raises(FinalReportValidationError, match="disabled until the LLM test artifacts"):
        run_final_report(trained_project)


def test_web_page_handles_missing_demo_selection(trained_project: ProjectPaths) -> None:
    client = create_app(trained_project.root).test_client()
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Manual input is enabled" in html
