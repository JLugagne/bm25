// Search example: a simple interactive document search using BM25.
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/JLugagne/bm25"
)

func main() {
	// A small collection of documents about programming languages.
	corpus := []string{
		"Go is a statically typed compiled language designed at Google",
		"Python is a high level interpreted language known for readability",
		"Rust is a systems programming language focused on safety and performance",
		"JavaScript is the language of the web used for both frontend and backend",
		"TypeScript adds static typing to JavaScript for larger codebases",
		"C is a low level language that provides direct memory access",
		"Java is a popular object oriented language that runs on the JVM",
		"Haskell is a purely functional programming language with strong static typing",
		"Ruby is a dynamic language designed for programmer happiness",
		"Elixir runs on the Erlang VM and is great for concurrent distributed systems",
		"Zig is a systems language intended as a better alternative to C",
		"Swift is developed by Apple for iOS and macOS application development",
		"Kotlin is a modern language for the JVM fully interoperable with Java",
		"Lua is a lightweight scripting language often embedded in applications",
		"Perl is known for its text processing capabilities and regular expressions",
	}

	tokenizer := func(s string) []string {
		return strings.Fields(strings.ToLower(s))
	}

	bm, err := bm25.NewBM25Okapi(corpus, tokenizer, 1.5, 0.75, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Indexed %d documents. Type a query (or 'quit' to exit):\n\n", bm.CorpusSize())

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "quit" || input == "exit" {
			break
		}

		query := tokenizer(input)
		topDocs, err := bm.GetTopN(query, 5)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			continue
		}

		scores, _ := bm.GetScores(query)

		// Print results with scores.
		fmt.Println()
		if len(topDocs) == 0 {
			fmt.Println("  No matching documents.")
		}
		for i, doc := range topDocs {
			// Find original index for the score.
			for j, c := range corpus {
				if strings.EqualFold(c, doc) {
					fmt.Printf("  %d. [%.4f] %s\n", i+1, scores[j], c)
					break
				}
			}
		}
		fmt.Println()
	}
}
