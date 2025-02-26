curl -L -X POST 'http://10.102.34.63:8080/v1/chat/completions' \
-H "Content-Type: application/json; charset=utf-8" \
-H 'accessToken: RN4xGIRM2i5CH1Hcogr3LRKDZf2Max76hdsMQmWn30B0hSEZn6KogPXTZuus3ScSgoqq2MVBIv1hV5OBNQsusujGVfvgcTSex0OWGj0LsDznVaNFBRyHM9aLStwkUEGB/ecF44ecoVWSO1aMI7jUipVI/hrBmQeFs0sGOYW5SU5uMbDpxE/iBmePEjWMC4BFxJEjfldhafRhO+8RurNMT1pwXtxI5g2ivXOe4uA+0oLtfLwbxITLerslmr5TjUUSu8ME9KFEBihMx6lxWYPPlFITdmmwyMPM4XzSJiAJjPqiVe3xZbElkYFZRgn+n7MtN1wxUAZiNu56bPsiIoba80as2x3wsa/TZaehtQWdmkqM60MmaApylDyiaZCG1o9tACEJpooLJGZrEg==' \
-d '{
  "model": "sdu",
  "messages": [
    {
      "role": "user",
      "content": "你叫什么名字？"
    }
  ],
  "tools": [],
  "do_sample": true,
  "temperature": 0,
  "top_p": 0,
  "n": 1,
  "max_tokens": 0,
  "stream": false
}'