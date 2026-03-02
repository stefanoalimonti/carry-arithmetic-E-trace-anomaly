CC = gcc
CFLAGS = -O3 -march=native -Wall
LDFLAGS = -lm

C_SOURCES = $(wildcard *.c)
BINARIES = $(C_SOURCES:.c=)

all: $(BINARIES)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINARIES)

.PHONY: all clean
