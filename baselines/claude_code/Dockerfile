FROM docker/sandbox-templates:claude-code

# Copy pre-downloaded Python packages (rank_bm25, whoosh, numpy)
# These are pre-installed on the host since Docker build lacks network access.
COPY pypackages/ /home/agent/.local/lib/python3.13/site-packages/

# Create mount points (need root for top-level dirs)
USER root
RUN mkdir -p /data /output && chown agent:agent /data /output
USER agent

# Copy scripts and prompts
WORKDIR /app
COPY scripts/ /app/scripts/
COPY CLAUDE.md /app/CLAUDE.md
COPY OUTPUT_SPEC.md /app/OUTPUT_SPEC.md

# Default entrypoint is claude
ENTRYPOINT ["claude"]
