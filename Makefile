CXX  = g++
CXXFLAGS    = -std=c++14 -static
INCLUDE   = -I./header
TARGETS   = run.exe
TARGETDIR = ./
SRCDIR   = ./source
OBJDIRS   = ./object
SOURCES   = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS   = $(subst $(SRCDIR),$(OBJDIRS), $(SOURCES:.cpp=.o))

$(TARGETS): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGETDIR)$@ $^

$(OBJDIRS)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

$(OBJDIRS)/%.o: $(INCLUDE)/%.hpp

all: clear $(TARGETS)

clear:
	rm $(TARGETDIR)$(TARGETS)
	rm $(OBJDIRS)/*.o
