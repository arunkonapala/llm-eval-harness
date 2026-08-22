from evaluator.models import MAX_RETRY_WAIT, parse_retry_wait

# Verbatim shapes of Groq's 429 body, minus the org id.
TPM = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.3-70b-versatile` in organization `org_x` service tier `on_demand` "
    "on tokens per minute (TPM): Limit 12000, Used 11800, Requested 500. "
    "Please try again in 1.5s."
)
TPD = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.3-70b-versatile` in organization `org_x` service tier `on_demand` "
    "on tokens per day (TPD): Limit 100000, Used 97492, Requested 3712. "
    "Please try again in 17m20.256s."
)


def test_per_minute_limit_is_retryable():
    assert parse_retry_wait(TPM) == 2.5  # 1.5s + 1s buffer


def test_per_day_limit_is_not_retryable():
    """The cap that actually broke CI: retrying can't refill a daily quota."""
    assert parse_retry_wait(TPD) is None


def test_requests_per_day_also_fatal():
    assert parse_retry_wait("on requests per day (RPD): Limit 1000. "
                            "Please try again in 2.0s.") is None


def test_long_wait_is_not_worth_sleeping_through():
    assert parse_retry_wait("Please try again in 5m0s.") is None


def test_wait_at_the_cap_is_still_retryable():
    secs = MAX_RETRY_WAIT - 1  # +1s buffer lands exactly on the cap
    assert parse_retry_wait(f"Please try again in {secs}s.") == MAX_RETRY_WAIT


def test_minutes_and_seconds_are_combined():
    assert parse_retry_wait("Please try again in 1m4.0s.") == 65.0


def test_unparseable_message_falls_back_to_a_short_sleep():
    assert parse_retry_wait("Rate limit reached. Slow down.") == 20.0
