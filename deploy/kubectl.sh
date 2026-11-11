# Shared kubectl wrappers. Source it, do not execute it:
#
#     k get pods -n credit-dev         # streams; stderr stays on stderr
#     POD="$(k_value get pod -o name)" # captures stdout; retries
#     ... | k_pipe apply -f -          # manifest on stdin; never retries
#
# SSL_CERT_FILE is dropped from all three: on this machine it points Go
# clients at a bundle that does not carry the local TLS-interception root.
#
# k_value retries because that interception drops established connections in
# bursts. x509 "unknown authority" is not transient -- bootstrap.sh resolves
# it -- and is not on the list.

K_RETRIES="${K_RETRIES:-6}"
K_RETRY_DELAY="${K_RETRY_DELAY:-10}"

_k_is_transient() {
  case "$1" in
    *wsarecv*|*wsasend*) return 0 ;;
    *"unexpected EOF"*) return 0 ;;
    *"TLS handshake timeout"*) return 0 ;;
    *"i/o timeout"*) return 0 ;;
    *) return 1 ;;
  esac
}

# For `rollout status` and `wait`, which stream progress for minutes.
k() {
  env -u SSL_CERT_FILE kubectl "$@"
}

# For everything else, captured or merely short.
k_value() {
  local attempt=1 out err rc errfile
  errfile="$(mktemp)"
  while :; do
    out="$(env -u SSL_CERT_FILE kubectl "$@" 2>"$errfile")"
    rc=$?
    err="$(cat "$errfile")"
    if [ "$rc" -eq 0 ] || [ "$attempt" -ge "$K_RETRIES" ] \
       || ! _k_is_transient "$err"; then
      rm -f "$errfile"
      [ -n "$err" ] && printf '%s\n' "$err" >&2
      [ -n "$out" ] && printf '%s\n' "$out"
      return "$rc"
    fi
    echo "   (transient transport failure; retry $attempt in ${K_RETRY_DELAY}s)" >&2
    sleep "$K_RETRY_DELAY"
    attempt=$((attempt + 1))
  done
}

# Piped stdin can only be read once, so this must never retry.
k_pipe() {
  env -u SSL_CERT_FILE kubectl "$@"
}
