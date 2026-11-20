# Writes a file as a C array, the way `xxd -i -n SYMBOL` does (used for the built-in models, so
# xxd is not needed). Usage: cmake -DINPUT=file -DOUTPUT=file.c -DSYMBOL=name -P embed_file.cmake
file(READ "${INPUT}" _hex HEX)
string(LENGTH "${_hex}" _hexlen)
math(EXPR _len "${_hexlen} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _body "${_hex}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){12})" "\\1\n  " _body "${_body}")
file(WRITE "${OUTPUT}"
    "unsigned char ${SYMBOL}[] = {\n  ${_body}\n};\nunsigned int ${SYMBOL}_len = ${_len};\n")
