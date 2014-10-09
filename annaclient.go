package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
)

func main() {
	filename := os.Args[1]
	buf, _ := ioutil.ReadFile(filename)
	input := []string{string(buf)}

	resp, _ := http.PostForm("http://localhost:8008", url.Values{"input": input})
	defer resp.Body.Close()
	body, _ := ioutil.ReadAll(resp.Body)
	fmt.Printf("%#v\n", body)

}
