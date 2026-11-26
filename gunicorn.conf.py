# Gunicorn reads this file from the working directory when the process is
# started without --config. Both Render start commands do that:
#   render.yaml: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
#   README:      gunicorn app:app --bind 0.0.0.0:$PORT --timeout 180
# Flags on the command line override this file, so bind, workers, and timeout
# stay whatever the Render start command already sets. Do not add
# --access-logformat there: that would replace the format below.

# Access lines keep the method, path, status, size, and duration.
# %(h)s is the client address and is omitted. %(r)s and %(q)s are omitted
# because a question can be carried in the query string (?q=).
access_log_format = "%(t)s %(m)s %(U)s %(s)s %(b)s %(M)sms"
