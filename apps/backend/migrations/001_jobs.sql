CREATE TABLE sessions (
    id uuid PRIMARY KEY,
    token_hash text NOT NULL UNIQUE,
    source text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE jobs (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES sessions (id),
    idempotency_key text NOT NULL,
    digest text NOT NULL,
    original text NOT NULL,
    felt text NOT NULL CHECK (felt IN ('auto', 'green', 'blue')),
    status text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 3),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    attempt_started_at timestamptz,
    completed_at timestamptz,
    available_at timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz,
    lease_token uuid,
    lease_expires_at timestamptz,
    result jsonb,
    overlay text,
    error text,
    UNIQUE (session_id, idempotency_key),
    CHECK ((status = 'processing') = (lease_token IS NOT NULL)),
    CHECK ((status = 'processing') = (lease_expires_at IS NOT NULL)),
    CHECK ((status IN ('completed', 'failed')) = (completed_at IS NOT NULL)),
    CHECK ((status = 'completed') = (result IS NOT NULL)),
    CHECK ((status = 'completed') = (overlay IS NOT NULL)),
    CHECK ((status = 'failed') = (error IS NOT NULL))
);

CREATE INDEX jobs_session_history ON jobs (session_id, created_at DESC, id DESC);
CREATE INDEX jobs_queued ON jobs (available_at, id) WHERE status = 'queued';
CREATE INDEX jobs_expired_leases ON jobs (lease_expires_at) WHERE status = 'processing';
