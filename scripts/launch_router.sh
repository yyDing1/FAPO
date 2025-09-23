unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
IP_LIST=(
    "xxxx"
    "xxxx"
    # ...
)
PORT=30000

declare -a WORKER_URLS
for ((i=0; i<${#IP_LIST[@]}; i++))
do
    IP=${IP_LIST[$i]}
    WORKER_URLS+=("http://$IP:$PORT")
done

printf '  %s\n' "${WORKER_URLS[@]}"
python -m sglang_router.launch_router --host 0.0.0.0 --port 30000  --balance-abs-threshold 4 --worker-urls "${WORKER_URLS[@]}" > router.log 2>&1 &
