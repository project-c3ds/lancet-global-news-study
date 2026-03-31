from eventregistry import *
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('NEWSAPI_KEY')
er = EventRegistry(apiKey=api_key)


def fetch_source_uri(url):
    """Fetch the source URI from EventRegistry for a given URL."""
    return er.getSourceUri(url)


def collect_recent_articles(source_uri, max_items=20,
                            date_start='2025-03-01', date_end='2026-03-06'):
    """Collect the most recent articles from a source (no keyword filter).
    Returns a dict with source metadata and the raw articles.
    Uses a 1-year date range to minimize token cost (5 tokens/year searched)."""
    q = QueryArticles(
        sourceUri=source_uri,
        dateStart=date_start,
        dateEnd=date_end,
    )
    q.setRequestedResult(RequestArticlesInfo(
        count=max_items,
        sortBy="date",
        sortByAsc=False,
        returnInfo=ReturnInfo(
            articleInfo=ArticleInfoFlags(
                body=True, title=True, url=True, authors=True,
                image=True, sentiment=True, concepts=True,
                categories=True, links=True, videos=True,
                socialScore=True, location=True, extractedDates=True,
                originalArticle=True, storyUri=True,
            )
        )
    ))
    res = er.execQuery(q)
    articles = res.get("articles", {}).get("results", [])

    # Extract metadata from collected articles
    lang_counts = Counter(a.get("lang", "unknown") for a in articles)
    most_recent_date = None
    if articles:
        dates = [a.get("dateTime") for a in articles if a.get("dateTime")]
        if dates:
            most_recent_date = max(dates)

    return {
        "articles_collected": len(articles),
        "articles_requested": max_items,
        "most_recent_date": most_recent_date,
        "language_counts": dict(lang_counts),
        "most_frequent_language": lang_counts.most_common(1)[0][0] if lang_counts else None,
        "articles": articles,
    }


def count_articles(source_uri, keywords=None, ignore_keywords=None,
                   date_start='2020-01-01', date_end='2026-03-06'):
    """Count articles matching keywords for a source without fetching them.
    Keywords should be a list of phrases -- they are OR'd together via QueryItems.OR().
    ignore_keywords is an optional list of phrases to exclude (also OR'd)."""
    kwargs = dict(
        sourceUri=source_uri,
        dateStart=date_start,
        dateEnd=date_end,
    )
    if keywords is not None:
        kwargs["keywords"] = QueryItems.OR(keywords)
        kwargs["keywordSearchMode"] = "phrase"
    if ignore_keywords is not None:
        kwargs["ignoreKeywords"] = QueryItems.OR(ignore_keywords)
        kwargs["ignoreKeywordSearchMode"] = "phrase"
    q = QueryArticlesIter(**kwargs)
    return q.count(er)


def get_usage():
    """Check API token usage and limits."""
    info = er.getUsageInfo()
    available = info.get("availableTokens", 0)
    used = info.get("usedTokens", 0)
    remaining = available - used
    print(f"Available: {available:,}")
    print(f"Used:      {used:,}")
    print(f"Remaining: {remaining:,}")
    return info
