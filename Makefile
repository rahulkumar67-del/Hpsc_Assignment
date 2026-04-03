CXX      = g++
CXXFLAGS = -Iinclude -O3 -Wall -std=c++17
SRC      = src/main.cpp src/physics.cpp
OBJ      = $(SRC:.cpp=.o)
TARGET   = particle_sim

all: results $(TARGET)

results:
	if not exist results mkdir results

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run individual tests
test_freefall: $(TARGET)
	./$(TARGET) freefall

test_constvel: $(TARGET)
	./$(TARGET) constvel

test_bounce: $(TARGET)
	./$(TARGET) bounce

test_errordt: $(TARGET)
	./$(TARGET) errordt

tests: $(TARGET)
	./$(TARGET) all

# Multi-particle runs
multi200: $(TARGET)
	./$(TARGET) multi200

multi1000: $(TARGET)
	./$(TARGET) multi1000

multi5000: $(TARGET)
	./$(TARGET) multi5000

clean:
	del /Q src\*.o $(TARGET).exe 2>nul || true