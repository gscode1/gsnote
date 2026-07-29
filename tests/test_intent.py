from app.intent import GENERAL, WHEN, detect_intent


def test_when_intent_detected():
    assert detect_intent("When did I capture that idea?") == WHEN
    assert detect_intent("What did I note last week?") == WHEN


def test_general_intent_default():
    assert detect_intent("What ideas do I have about note apps?") == GENERAL
    assert detect_intent("Tell me about my project notes") == GENERAL
