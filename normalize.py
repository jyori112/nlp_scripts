#!/usr/bin/env python3

import argparse
import re, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--normalize', '-n', nargs='+', choices=['lower', 'html', 'url', 'number'])
    parser.add_argument('--lower', '-l', action='store_true')
    parser.add_argument('--html', '-html', action='store_true')
    parser.add_argument('--url', '-url', action='store_true')
    parser.add_argument('--number', '-num', action='store_true')
    parser.add_argument('--empty_line', '-emp', action='store_true')
    parser.add_argument('--keep_decimal', '-d', action='store_true')

    args = parser.parse_args()

    if args.input:
        input_file = open(args.input)
    else:
        input_file = sys.stdin

    if args.output:
        output_file = open(args.output, 'w')
    else:
        output_file = sys.stdout

    html = re.compile('<.*?>')
    url = re.compile('https?://[\w/:%#\$&\?\(\)~\.=\+\-]+')

    if args.keep_decimal:
        number = re.compile('[0-9]+')
    else:
        number = re.compile('[0-9]+(\.[0-9]+)?')

    for line in input_file:
        line = line.rstrip()

        if args.lower:
            line = line.lower()

        if args.html:
            line = html.sub('', line)

        if args.url:
            line = url.sub('', line)

        if args.number:
            line = number.sub('0', line)

        if not args.empty_line or line:
            print(line, file=output_file)


if __name__ == '__main__':
    main()
