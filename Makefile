# Makefile for ccont tool

CC = $(CROSS_COMPILE)gcc
DEFINES=

CFLAGS = -O2 -Wall -lpthread -lnuma -lm -lrt

all: ccont
%: %.c
	$(CC) $(DEFINES) -o $@ $^ $(CFLAGS)

clean:
	$(RM) ccont *~
