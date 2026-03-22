// Benchmark tokenization throughput on lines from stdin.
//
// Usage:
//
//	cat file.txt | go run ./examples/benchmark _testdata/spm.model
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

	// Read all lines into memory so we can run multiple passes.
	var lines []string
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Loaded %d lines\n\n", len(lines))

	var bestTPS float64
	var bestTokens int
	var bestElapsed time.Duration

	for pass := 0; pass < 3; pass++ {
		label := fmt.Sprintf("Pass %d (measured)", pass)
		if pass == 0 {
			label = "Pass 0 (warmup)"
		}
		totalTokens := 0
		start := time.Now()
		for _, line := range lines {
			ids, _ := tok.Encode(line)
			totalTokens += len(ids)
		}
		elapsed := time.Since(start)
		tps := float64(totalTokens) / elapsed.Seconds()
		fmt.Printf("%-20s  %d tokens in %v  (%.0f tok/s, %.0f lines/s)\n",
			label, totalTokens, elapsed.Round(time.Microsecond), tps, float64(len(lines))/elapsed.Seconds())

		if pass > 0 && tps > bestTPS {
			bestTPS = tps
			bestTokens = totalTokens
			bestElapsed = elapsed
		}
	}

	fmt.Printf("\nBest: %d lines, %d tokens in %v — %.0f tokens/sec, %.0f lines/sec\n",
		len(lines), bestTokens, bestElapsed.Round(time.Microsecond), bestTPS, float64(len(lines))/bestElapsed.Seconds())
}
