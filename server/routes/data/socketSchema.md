Quote
T	string	message type, always “q”
S	string	symbol
ax	string	ask exchange code
ap	number	ask price
as	int	ask size in round lots
bx	string	bid exchange code
bp	number	bid price
bs	int	bid size in round lots
c	array	quote condition
t	string	RFC-3339 formatted timestamp with nanosecond precision
z	string	tape

Example: 
```json
{
    "T": "t",
    "i": 96921,
    "S": "AAPL",
    "x": "D",
    "p": 126.55,
    "s": 1,
    "t": "2021-02-22T15:51:44.208Z",
    "c": ["@", "I"],
    "z": "C"
  }
```


Bars
Attribute	Type	Description
T	string	message type: “b”, “d” or “u”
S	string	symbol
o	number	open price
h	number	high price
l	number	low price
c	number	close price
v	int	volume
t	string	RFC-3339 formatted timestamp

Example:
```json
{
  "T": "b",
  "S": "SPY",
  "o": 388.985,
  "h": 389.13,
  "l": 388.975,
  "c": 389.12,
  "v": 49378,
  "t": "2021-02-22T19:15:00Z"
}
```

