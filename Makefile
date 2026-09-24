CC = gcc

LIBS = -Wall -Wextra

SRC = 5.c
TARGET = a.exe

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(LIBS)

clean:
	del $(TARGET)