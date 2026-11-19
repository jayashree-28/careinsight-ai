
from google.adk.tools import ToolContext


def validate_medication_report(report_data: dict) -> dict:
    """
    Deterministically validate the consistency and completeness
    of a CareInsight AI medication experience evidence package.
    """

    errors = []
    warnings = []
    checks = []

    def record_check(name: str, passed: bool, details: str):
        checks.append({
            "name": name,
            "status": "PASSED" if passed else "FAILED",
            "details": details,
        })

        if not passed:
            errors.append(f"{name}: {details}")

    def record_warning(name: str, details: str):
        checks.append({
            "name": name,
            "status": "WARNING",
            "details": details,
        })
        warnings.append(f"{name}: {details}")

    # 1. Validate report status and required sections

    if report_data.get("report_status") != "evidence_collected":
        record_check(
            "Evidence collection",
            False,
            "Report evidence was not collected successfully.",
        )
    else:
        record_check(
            "Evidence collection",
            True,
            "Evidence package was collected successfully.",
        )

    retrieval = report_data.get("retrieval_analysis") or {}
    sentiment = report_data.get("sentiment_analysis") or {}
    topics = report_data.get("topic_analysis") or {}

    required_sections = {
        "Retrieval analysis": retrieval,
        "Sentiment analysis": sentiment,
        "Topic analysis": topics,
    }

    for section_name, section_data in required_sections.items():
        record_check(
            section_name,
            bool(section_data),
            (
                "Section is present."
                if section_data
                else "Required analysis section is missing."
            ),
        )

    if errors:
        return {
            "status": "REVISION_REQUIRED",
            "checks_passed": sum(
                check["status"] == "PASSED" for check in checks
            ),
            "checks_total": len(checks),
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "summary": "Required evidence is missing.",
        }

    # 2. Validate retrieval and sentiment review counts

    retrieval_count = retrieval.get("total_reviews")
    sentiment_count = sentiment.get("total_reviews_analyzed")
    topic_count = topics.get("reviews_analyzed")

    valid_counts = all(
        isinstance(value, int) and not isinstance(value, bool)
        and value >= 0
        for value in (
            retrieval_count,
            sentiment_count,
            topic_count,
        )
    )

    if not valid_counts:
        record_check(
            "Review count validity",
            False,
            "One or more review counts are missing or invalid.",
        )
    else:
        record_check(
            "Review count validity",
            True,
            "All review counts are non-negative integers.",
        )

        record_check(
            "Retrieval and sentiment count consistency",
            retrieval_count == sentiment_count,
            (
                f"Both analyses contain {retrieval_count} reviews."
                if retrieval_count == sentiment_count
                else (
                    f"Retrieval contains {retrieval_count} reviews, "
                    f"while sentiment contains {sentiment_count}."
                )
            ),
        )

        if topic_count == sentiment_count:
            record_check(
                "Topic and sentiment count consistency",
                True,
                "Both analyses cover the same number of reviews.",
            )
        else:
            record_warning(
                "Topic and sentiment count consistency",
                (
                    f"Topic analysis covers {topic_count} reviews, "
                    f"while sentiment analysis covers "
                    f"{sentiment_count}. Topic analysis may use "
                    "a subset of reviews."
                ),
            )

    # 3. Validate rating distribution and average rating

    rating_distribution = retrieval.get("rating_distribution")
    average_rating = retrieval.get("average_rating")

    if not isinstance(rating_distribution, dict) or not rating_distribution:
        record_check(
            "Rating distribution",
            False,
            "Rating distribution is missing or invalid.",
        )
    else:
        try:
            rating_counts = {
                int(rating): count
                for rating, count in rating_distribution.items()
            }

            valid_rating_values = (
                all(1 <= rating <= 10 for rating in rating_counts)
                and all(
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                    for count in rating_counts.values()
                )
            )

            if not valid_rating_values:
                raise ValueError("Invalid rating values or counts.")

            rating_total = sum(rating_counts.values())

            record_check(
                "Rating distribution",
                rating_total == retrieval_count,
                (
                    f"Rating counts sum to {rating_total}, "
                    f"matching {retrieval_count} retrieved reviews."
                    if rating_total == retrieval_count
                    else (
                        f"Rating counts sum to {rating_total}, "
                        f"but retrieval contains {retrieval_count}."
                    )
                ),
            )

            if rating_total > 0 and isinstance(
                average_rating, (int, float)
            ):
                calculated_average = sum(
                    rating * count
                    for rating, count in rating_counts.items()
                ) / rating_total

                average_matches = (
                    abs(calculated_average - average_rating) <= 0.011
                )

                record_check(
                    "Average rating",
                    average_matches,
                    (
                        f"Reported average: {average_rating}; "
                        f"calculated average: "
                        f"{calculated_average:.2f}."
                    ),
                )
            else:
                record_check(
                    "Average rating",
                    False,
                    "Average rating is missing or cannot be calculated.",
                )

        except (ValueError, TypeError):
            record_check(
                "Rating distribution",
                False,
                "Rating distribution contains invalid values.",
            )

    # 4. Validate sentiment distribution

    distribution = sentiment.get("sentiment_distribution")

    if not isinstance(distribution, dict) or not distribution:
        record_check(
            "Sentiment distribution",
            False,
            "Sentiment distribution is missing or invalid.",
        )
    else:
        try:
            expected_labels = {"positive", "neutral", "negative"}

            valid_labels = set(distribution.keys()) == expected_labels

            valid_values = all(
                isinstance(item, dict)
                and isinstance(item.get("count"), int)
                and not isinstance(item.get("count"), bool)
                and item["count"] >= 0
                and isinstance(item.get("percentage"), (int, float))
                and not isinstance(item.get("percentage"), bool)
                for item in distribution.values()
            )

            if not valid_labels or not valid_values:
                raise ValueError("Invalid sentiment labels or values.")

            sentiment_total = sum(
                item["count"] for item in distribution.values()
            )

            record_check(
                "Sentiment count reconciliation",
                sentiment_total == sentiment_count,
                (
                    f"Sentiment counts sum to {sentiment_total}, "
                    f"matching {sentiment_count} analyzed reviews."
                    if sentiment_total == sentiment_count
                    else (
                        f"Sentiment counts sum to {sentiment_total}, "
                        f"but {sentiment_count} reviews were analyzed."
                    )
                ),
            )

            percentage_total = sum(
                item["percentage"] for item in distribution.values()
            )

            record_check(
                "Sentiment percentage reconciliation",
                abs(percentage_total - 100.0) <= 0.1,
                (
                    f"Sentiment percentages sum to "
                    f"{percentage_total:.2f}%."
                ),
            )

            percentage_matches_counts = all(
                abs(
                    item["percentage"]
                    - item["count"] / sentiment_count * 100
                ) <= 0.02
                for item in distribution.values()
            ) if sentiment_count else False

            record_check(
                "Sentiment percentage accuracy",
                percentage_matches_counts,
                "Reported percentages agree with calculated percentages."
                if percentage_matches_counts
                else "Reported percentages do not match calculated percentages.",
            )

        except (ValueError, TypeError, ZeroDivisionError):
            record_check(
                "Sentiment distribution validity",
                False,
                "Sentiment distribution contains invalid values.",
            )
    # 5. Validate topic counts and outliers

    topic_list = topics.get("topics")
    outliers = topics.get("outlier_reviews")
    number_of_topics = topics.get("number_of_topics")

    # Topic modeling may be intentionally skipped when there
    # are fewer than 20 reviews.
    if (
        isinstance(topic_list, list)
        and len(topic_list) == 0
        and outliers is None
        and number_of_topics == 0
        and topic_count < 20
    ):
        record_warning(
            "Topic modeling skipped",
            (
                f"Topic modeling was skipped because only "
                f"{topic_count} reviews were available. "
                "No topic counts or outlier counts were calculated."
            ),
        )

    elif (
        not isinstance(topic_list, list)
        or not isinstance(outliers, int)
        or isinstance(outliers, bool)
        or outliers < 0
        or not isinstance(number_of_topics, int)
        or isinstance(number_of_topics, bool)
        or number_of_topics < 0
    ):
        record_check(
            "Topic count reconciliation",
            False,
            "Topic list, topic count, or outlier count is missing or invalid.",
        )

    else:
        valid_topic_counts = all(
            isinstance(topic, dict)
            and isinstance(topic.get("review_count"), int)
            and not isinstance(topic.get("review_count"), bool)
            and topic["review_count"] >= 0
            for topic in topic_list
        )

        if not valid_topic_counts:
            record_check(
                "Topic count reconciliation",
                False,
                "One or more topic review counts are invalid.",
            )

        elif number_of_topics != len(topic_list):
            record_warning(
                "Topic count reconciliation",
                (
                    f"Only {len(topic_list)} topic records are returned, "
                    f"but the model reports {number_of_topics} topics. "
                    "Complete topic reconciliation cannot be confirmed."
                ),
            )

        else:
            assigned_reviews = sum(
                topic["review_count"] for topic in topic_list
            )

            total_topic_reviews = assigned_reviews + outliers

            record_check(
                "Topic count reconciliation",
                total_topic_reviews == topic_count,
                (
                    f"Assigned reviews ({assigned_reviews}) plus "
                    f"outliers ({outliers}) equal {topic_count}."
                    if total_topic_reviews == topic_count
                    else (
                        f"Assigned reviews plus outliers equal "
                        f"{total_topic_reviews}, but topic analysis "
                        f"contains {topic_count} reviews."
                    )
                ),
            )


    # 6. Check supporting review availability

    retrieval_examples = retrieval.get("supporting_reviews", [])
    sentiment_examples = sentiment.get("supporting_reviews", [])

    if retrieval_examples and sentiment_examples:
        record_check(
            "Supporting review availability",
            True,
            "Supporting examples are present in retrieval and sentiment results.",
        )
    else:
        record_warning(
            "Supporting review availability",
            "Supporting examples are missing from one or more analysis sections.",
        )

    # 7. Final validation decision

    status = "APPROVED" if not errors else "REVISION_REQUIRED"

    checks_passed = sum(
        check["status"] == "PASSED" for check in checks
    )

    return {
        "status": status,
        "checks_passed": checks_passed,
        "checks_total": len(checks),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "summary": (
            "Evidence passed all required deterministic checks."
            if status == "APPROVED"
            else "Evidence requires correction before presentation."
        ),
    }


def validate_report_evidence(tool_context: ToolContext) -> dict:
    """
    Validate the exact evidence package generated by
    the Report Agent, without rerunning the analyses.
    """

    report_data = tool_context.state.get("careinsight_evidence")

    if not report_data:
        return {
            "status": "REVISION_REQUIRED",
            "checks_passed": 0,
            "checks_total": 0,
            "checks": [],
            "errors": [
                "Shared report evidence was not found in session state."
            ],
            "warnings": [],
            "summary": "Unable to validate report evidence.",
        }

    return validate_medication_report(report_data)
