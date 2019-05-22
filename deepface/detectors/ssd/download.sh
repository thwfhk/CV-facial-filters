echo "download model graph : Detectors"

DIR="$(cd "$(dirname "$0")" && pwd)"

extract_download_url() {
        if ! [ -x "$(command -v wget)" ]; then
          echo 'Error: wget is not installed.' >&2
          exit 1
        fi
        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

extract_filename() {
        echo "$DIR/${1##*/}"
}

download_mediafire() {
        if ! [ -x "$(command -v curl)" ]; then
          echo 'Error: curl is not installed.' >&2
          exit 1
        fi
        curl -L -o $( extract_filename $1 ) -C - $( extract_download_url $1 )
}

$( download_mediafire http://www.mediafire.com/file/ivwws1znd4y2v9y/graph_mobilenet_v2_fddb_180627.pb )
$( download_mediafire http://www.mediafire.com/file/a04pe6qzlevsso8/graph_mobilenet_v2_all_180627.pb )
