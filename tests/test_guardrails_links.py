"""Questions with web links are refused before any lesson is made."""

import pytest

from lesson_platform.guardrails import check_question


@pytest.mark.parametrize("question", [
    "* * * SCAM SERVICE 24/7: https://myxa.cc/ * * * hs=f003ffb2612e00711fdc96731ccaae95*",
    "visit http://example.com now",
    "go to www.cheap-stuff.net",
    "best deals at spam-site.xyz/offer",
    "BUY NOW AT HTTPS://SPAM.RU",
])
def test_links_are_refused(question):
    assert check_question(question) == "Please ask your question without a web link or website address."


@pytest.mark.parametrize("question", [
    "Why is the sky blue?",
    "How does the internet work?",
    "What is 3.14 used for?",
    "How do websites load so fast?",
    "Why does Mr. Smith's car make noise?",
])
def test_normal_questions_still_pass(question):
    assert check_question(question) is None
