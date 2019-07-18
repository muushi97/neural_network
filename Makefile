# ---------------------------------------------------------------------- #
#                                                                        #
#             __  __       _        _____ _ _                            #
#            |  \/  | __ _| | _____|  ___(_) | ___                       #
#            | |\/| |/ _` | |/ / _ \ |_  | | |/ _ \                      #
#            | |  | | (_| |   <  __/  _| | | |  __/                      #
#            |_|  |_|\__,_|_|\_\___|_|   |_|_|\___|                      #
#                                                                        #
# ---------------------------------------------------------------------- #

# コンパイラ
CXX        = g++
#CXX        = clang++
# コンパイラオプション
CXXFLAGS   = -std=c++17 -static -g -O0
#CXXFLAGS   = -std=c++17 -static -g -O3 -mtune=native -march=native
# インクルードディレクトリ
INCLUDE    = -I./inc
# 出力ファイル名
TARGETS    = run.momiage
# 出力ディレクトリ
TARGETDIR  = .
# ソースのルートディレクトリ
SRCROOT    = ./src
# オブジェクトファイルのルートディレクトリ
OBJROOT    = ./obj
# ソースファイルが直下にあるディレクトリたち
SRCDIRS    = $(shell find $(SRCROOT) -type d)
# ソースファイルたち
SOURCES    = $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.cpp))
# オブジェクトファイルたち
OBJECTS    = $(subst $(SRCROOT), $(OBJROOT), $(SOURCES:.cpp=.o))
# オブジェクトファイルが直下にあるディレクトリたち
OBJDIRS    = $(subst $(SRCROOT), $(OBJROOT), $(SRCDIRS))
# 依存関係ファイル
DEPENDS    = $(OBJECTS:.o=.d)


# リンク
$(TARGETS): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGETDIR)/$@ $^

# 依存関係からビルド
-include $(DEPENDS)

# オブジェクトファイルほしい
$(OBJROOT)/%.o: $(SRCROOT)/%.cpp
	@if [ ! -e $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CXX) $(CXXFLAGS) -MMD -MP $(INCLUDE) -o $@ -c $<

# 篠沢教授に全部
all: clean $(TARGETS)

# さよならバイバイ
clean:
	- rm $(addsuffix /*.o, $(OBJDIRS))
	- rm $(addsuffix /*.d, $(OBJDIRS))
	- rm $(TARGETDIR)/$(TARGETS)

run: $(TARGETS)
	- ./$(TARGETS)

# こいつらファイルじゃない
.PHONY: all clean

