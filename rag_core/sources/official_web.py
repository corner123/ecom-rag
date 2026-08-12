"""Whitelisted, read-only collector for official web documentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from http.client import RemoteDisconnected
import re
import time
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import CollectedSource
from .schema import DocumentRecord, SourceRecord, canonical_hash, content_hash, utc_now


@dataclass(slots=True)
class FetchResponse:
    url: str
    body: bytes | str
    content_type: str = "text/html"
    charset: str = "utf-8"
    etag: str | None = None
    last_modified: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


Fetcher = Callable[[str], FetchResponse]


class OfficialWebSource:
    """Fetch documentation only from an explicit official-domain allowlist.

    Tests and offline jobs may inject ``fetcher``.  The default implementation
    uses only the standard library, validates redirects, and enforces a response
    size limit.
    """

    def __init__(
        self,
        source_id: str,
        urls: Iterable[str],
        *,
        allowed_domains: Iterable[str],
        version: str = "unversioned",
        license: str = "see-source-terms",
        fetcher: Fetcher | None = None,
        timeout: float = 20.0,
        max_bytes: int = 5_000_000,
        fetch_retries: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.source_id = source_id
        self.urls = tuple(_canonical_url(url) for url in urls)
        self.allowed_domains = tuple(domain.strip().lower().rstrip(".") for domain in allowed_domains)
        if not self.urls:
            raise ValueError("OfficialWebSource requires at least one URL")
        if not self.allowed_domains:
            raise ValueError("OfficialWebSource requires allowed_domains")
        self.version = version
        self.license = license
        self.fetcher = fetcher or self._fetch
        self.timeout = timeout
        self.max_bytes = max_bytes
        if fetch_retries < 1 or fetch_retries > 5:
            raise ValueError("fetch_retries must be between 1 and 5")
        self.fetch_retries = int(fetch_retries)
        self.metadata = dict(metadata or {})
        for url in self.urls:
            self._validate_url(url)

    def collect(self) -> CollectedSource:
        fetched_at = utc_now()
        documents: list[DocumentRecord] = []
        for requested_url in self.urls:
            self._validate_url(requested_url)
            response = self.fetcher(requested_url)
            final_url = _canonical_url(response.url or requested_url)
            self._validate_url(final_url)
            text = self._decode_body(response)
            media_type = response.content_type.split(";", 1)[0].strip().lower()
            if media_type in {"text/html", "application/xhtml+xml"}:
                cleaned, title = clean_html_document(text)
                language = "html"
                output_media_type = "text/markdown"
            else:
                cleaned = _normalize_text(text)
                title = final_url.rstrip("/").rsplit("/", 1)[-1] or final_url
                language = "text"
                output_media_type = media_type or "text/plain"
            if not cleaned:
                continue
            documents.append(
                DocumentRecord(
                    source_id=self.source_id,
                    relative_path=final_url,
                    content=cleaned,
                    media_type=output_media_type,
                    title=title,
                    language=language,
                    metadata={
                        "record_kind": "official_web_page",
                        "requested_url": requested_url,
                        "canonical_url": final_url,
                        "version": self.version,
                        "license": self.license,
                        "fetched_at": fetched_at,
                        "etag": response.etag,
                        "last_modified": response.last_modified,
                        "response_headers": response.headers,
                        "file_type": "markdown" if output_media_type == "text/markdown" else "text",
                        "content_format": "markdown" if output_media_type == "text/markdown" else "text",
                        "parser_backend": "stdlib-html-cleaner" if language == "html" else "stdlib-text",
                        "parser_version": "1.0",
                    },
                )
            )

        documents.sort(key=lambda item: item.relative_path)
        aggregate = canonical_hash(
            [(document.relative_path, document.content_hash) for document in documents]
        )
        record = SourceRecord(
            source_id=self.source_id,
            source_type="official_web",
            uri=f"official-web://{self.source_id}",
            version=self.version,
            license=self.license,
            fetched_at=fetched_at,
            content_hash=aggregate,
            metadata={
                **self.metadata,
                "urls": list(self.urls),
                "allowed_domains": list(self.allowed_domains),
                "document_count": len(documents),
            },
        )
        return CollectedSource(record=record, documents=documents)

    def _validate_url(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https":
            raise ValueError(f"official URL must use HTTPS: {url}")
        if parsed.username or parsed.password:
            raise ValueError(f"credentials are not allowed in source URLs: {url}")
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if not hostname or not any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in self.allowed_domains
        ):
            raise ValueError(f"URL is outside the official-domain allowlist: {url}")

    def _fetch(self, url: str) -> FetchResponse:
        request = Request(
            url,
            headers={
                "User-Agent": "ai-team-knowledge-base-ingestion/1.0",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            },
        )
        for attempt in range(1, self.fetch_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - allowlist enforced
                    final_url = response.geturl()
                    self._validate_url(final_url)
                    body = response.read(self.max_bytes + 1)
                    if len(body) > self.max_bytes:
                        raise ValueError(f"response exceeds max_bytes={self.max_bytes}: {url}")
                    headers = response.headers
                    return FetchResponse(
                        url=final_url,
                        body=body,
                        content_type=headers.get_content_type() or "application/octet-stream",
                        charset=headers.get_content_charset() or "utf-8",
                        etag=headers.get("ETag"),
                        last_modified=headers.get("Last-Modified"),
                        headers={key.lower(): value for key, value in headers.items()},
                    )
            except HTTPError as exc:
                # Client errors are deterministic catalog/configuration issues.
                if exc.code < 500 or attempt == self.fetch_retries:
                    raise
            except (RemoteDisconnected, URLError, TimeoutError, ConnectionError, OSError):
                if attempt == self.fetch_retries:
                    raise
            time.sleep(0.4 * attempt)
        raise AssertionError("unreachable fetch retry state")

    @staticmethod
    def _decode_body(response: FetchResponse) -> str:
        if isinstance(response.body, str):
            return response.body
        try:
            return response.body.decode(response.charset or "utf-8")
        except (LookupError, UnicodeDecodeError):
            return response.body.decode("utf-8", errors="replace")


class _MainContentParser(HTMLParser):
    _ignored = {"script", "style", "noscript", "svg", "nav", "header", "footer", "form"}
    _void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    _blocks = {"p", "div", "section", "article", "main", "aside", "blockquote", "pre", "table", "tr", "ul", "ol", "dl"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.all_parts: list[str] = []
        self.main_parts: list[str] = []
        self.title_parts: list[str] = []
        self.main_depth = 0
        # Remember which concrete open element started a main-content scope.
        # Comparing tag names alone truncates pages that nest repeated ``div``
        # elements (as the Python documentation does).
        self.open_tags: list[tuple[str, bool]] = []
        self.skip_depth = 0
        self.title_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self.skip_depth:
            if tag not in self._void:
                self.skip_depth += 1
            return
        if tag in self._ignored:
            self.skip_depth = 1
            return
        if tag == "title":
            self.title_depth += 1
            return
        is_main = tag in {"main", "article"} or any(
            key.lower() == "role" and (value or "").lower() == "main" for key, value in attrs
        )
        if is_main:
            self.main_depth += 1
        if tag not in self._void:
            self.open_tags.append((tag, is_main))
        if tag in self._blocks:
            self._emit("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._emit("\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._emit("\n- ")
        elif tag == "br":
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if tag == "title":
            self.title_depth = max(0, self.title_depth - 1)
            return
        if tag in self._blocks or tag in {"li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._emit("\n")
        if self.open_tags:
            match = next(
                (
                    index
                    for index in range(len(self.open_tags) - 1, -1, -1)
                    if self.open_tags[index][0] == tag
                ),
                None,
            )
            if match is not None:
                closed = self.open_tags[match:]
                del self.open_tags[match:]
                self.main_depth = max(
                    0,
                    self.main_depth
                    - sum(1 for _, starts_main in closed if starts_main),
                )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "hr"} and not self.skip_depth:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.title_depth:
            self.title_parts.append(data)
            return
        self._emit(data)

    def _emit(self, value: str) -> None:
        self.all_parts.append(value)
        if self.main_depth:
            self.main_parts.append(value)


def clean_html_document(html: str) -> tuple[str, str]:
    """Extract readable main/article text, with a whole-page fallback."""

    parser = _MainContentParser()
    parser.feed(html)
    parser.close()
    main = _normalize_text("".join(parser.main_parts))
    fallback = _normalize_text("".join(parser.all_parts))
    title = _normalize_inline(" ".join(parser.title_parts))
    content = main if main else fallback
    if title and not content.startswith(f"# {title}"):
        content = f"# {title}\n\n{content}" if content else f"# {title}"
    return content, title


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    port = parsed.port
    netloc = hostname
    if port and not (scheme == "https" and port == 443):
        netloc = f"{hostname}:{port}"
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def _normalize_inline(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
