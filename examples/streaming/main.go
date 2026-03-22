// Tokenize lines from stdin, printing per-line and cumulative stats.
//
// Usage:
//
//	echo "line1\nline2" | go run ./examples/streaming _testdata/spm.model
//	go run ./examples/streaming _testdata/spm.model < file.txt
package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"time"

	sp "github.com/tggo/goSentencePiece"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <model>\n", os.Args[0])
		os.Exit(1)
	}

	tok, err := sp.NewTokenizer(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var totalLines, totalTokens int
	start := time.Now()

	for scanner.Scan() {
		line := scanner.Text()
		totalLines++
		ids, _ := tok.Encode(line)
		n := len(ids)
		totalTokens += n

		preview := line
		if len(preview) > 50 {
			preview = preview[:50] + "..."
		}
		fmt.Printf("line %4d | %4d tokens | %s\n", totalLines, n, preview)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	elapsed := time.Since(start)
	avg := 0.0
	if totalLines > 0 {
		avg = float64(totalTokens) / float64(totalLines)
	}
	tps := 0.0
	if elapsed.Seconds() > 0 {
		tps = float64(totalTokens) / elapsed.Seconds()
	}

	fmt.Println("\n--- Summary ---")
	fmt.Printf("Lines:        %d\n", totalLines)
	fmt.Printf("Total tokens: %d\n", totalTokens)
	fmt.Printf("Avg tok/line: %.1f\n", avg)
	fmt.Printf("Elapsed:      %v\n", elapsed.Round(time.Microsecond))
	fmt.Printf("Tokens/sec:   %.0f\n", tps)
}
