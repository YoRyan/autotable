# first stage
FROM python:3 AS builder
WORKDIR /code
COPY . .
RUN pip install --user --no-cache-dir --use-feature=in-tree-build --no-warn-script-location .

# second stage
FROM python:3-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
  libspatialindex-dev
RUN groupadd -r autotable && useradd --no-log-init -r -g autotable autotable
USER autotable
COPY --from=builder --chown=autotable:autotable /root/.local/ /home/autotable/.local/
ENV PATH=/home/autotable/.local/bin/:$PATH
ENTRYPOINT ["autotable"]