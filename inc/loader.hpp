#ifndef IG_LOADER_HPP
#define IG_LOADER_HPP

#include <fstream>
#include <string>

#include "tensor.hpp"

// ../../dataset/mnist/

//#define GAZOU_GA_HOSHI(image, q) do { ¥
//    fseek(image, 16 + 28 * 28 * q, SEEK_SET); ¥
//        for (j = 0; j < 28; j++) { ¥
//            for (i = 0; i < 28; i++) { ¥
//                fread(&buf, sizeof(buf), 1, image); ¥
//                pixel[i][j] = buf; ¥
//                printf("%02x", pixel[i][j]); ¥
//            } ¥
//            printf("¥n"); ¥
//        } ¥
//    } while (0)
//
//// 画像取得マクロ
//#define RABERU_GA_HOSHI(image, f) do { ¥
//        fseek(image, 8 + f, SEEK_SET); ¥
//        fread(&buf, sizeof(buf), 1, image); ¥
//    } while (0)

template <class T>
class loader {
public:
    virtual tensor<T> get_train_date(std::size_t i) = 0;
    virtual tensor<T> get_train_label(std::size_t i) = 0;
    virtual tensor<T> get_test_date(std::size_t i) = 0;
    virtual tensor<T> get_test_label(std::size_t i) = 0;
    virtual tensor<T> get_valid_date(std::size_t i) = 0;
    virtual tensor<T> get_valid_label(std::size_t i) = 0;

    virtual std::size_t train_size() const = 0;
    virtual std::size_t test_size() const = 0;
    virtual std::size_t valid_size() const = 0;

    virtual ~loader() { }
};

template <class T>
class mnist_loader : public loader<T> {
    std::ifstream train_image;
    std::ifstream train_label;
    std::ifstream test_image;
    std::ifstream test_label;

    double data_max, data_min;
    std::size_t train_size_;
    std::size_t test_size_;
    std::size_t valid_size_;
    std::size_t width, height;

    std::size_t convert_bytes_to_integer(char *buf, std::size_t begin, std::size_t end) {
        std::size_t c = 0;
        if (end - begin > 0) {
            for (int i = begin; i < end; i++)
                c = c << 8 | (unsigned char)buf[i];
        }
        else {
            for (int i = end; i < begin; i++)
                c = c << 8 | (unsigned char)buf[end - i + begin];
        }
        return c;
    }

    // 先頭の絶対位置
    std::size_t image_seek_size(std::size_t i, std::size_t w, std::size_t h) const { return 16 + width * height * i; }
    std::size_t label_seek_size(std::size_t i) const { return 8 + i; }

    // 画像読み込み
    // ラベル読み込み

public:
    mnist_loader(std::string train_image, std::string train_label, std::string test_image, std::string test_label, double min, double max)
            : train_image(train_image, std::ios::binary), train_label(train_label, std::ios::binary)
            , test_image(test_image, std::ios::binary), test_label(test_label, std::ios::binary) 
            , data_max(max), data_min(min) {
        char buf[16];

        this->train_image.read(buf, 16);
        if (convert_bytes_to_integer(buf, 0, 4) != 2051) return;
        train_size_ = convert_bytes_to_integer(buf, 4, 8);
        width = convert_bytes_to_integer(buf, 8, 12);
        height = convert_bytes_to_integer(buf, 12, 16);
        this->train_image.seekg(0, std::ios::end);
        if (this->train_image.tellg() != 16 + train_size_ * width * height) return;

        this->train_label.read(buf, 8);
        if (convert_bytes_to_integer(buf, 0, 4) != 2049) return;
        if (train_size_ != convert_bytes_to_integer(buf, 4, 8)) return;
        this->train_label.seekg(0, std::ios::end);
        if (this->train_label.tellg() != 8 + train_size_) return;

        this->test_image.read(buf, 16);
        if (convert_bytes_to_integer(buf, 0, 4) != 2051) return;
        test_size_ = convert_bytes_to_integer(buf, 4, 8);
        if (width != convert_bytes_to_integer(buf, 8, 12)) return;
        if (height != convert_bytes_to_integer(buf, 12, 16)) return;
        this->test_image.seekg(0, std::ios::end);
        if (this->test_image.tellg() != 16 + test_size_ * width * height) return;

        this->test_label.read(buf, 8);
        if (convert_bytes_to_integer(buf, 0, 4) != 2049) return;
        if (test_size_ != convert_bytes_to_integer(buf, 4, 8)) return;
        this->test_label.seekg(0, std::ios::end);
        if (this->test_label.tellg() != 8 + test_size_) return;

        valid_size_ = 0;
    }

    virtual tensor<T> get_train_date(std::size_t i) {
        return tensor<T>{1};
    }
    virtual tensor<T> get_train_label(std::size_t i) {
        return tensor<T>{1};
    }
    virtual tensor<T> get_test_date(std::size_t i) {
        return tensor<T>{1};
    }
    virtual tensor<T> get_test_label(std::size_t i) {
        return tensor<T>{1};
    }
    virtual tensor<T> get_valid_date(std::size_t i) {
        return tensor<T>{1};
    }
    virtual tensor<T> get_valid_label(std::size_t i) {
        return tensor<T>{1};
    }

    virtual std::size_t train_size() const { return train_size_; }
    virtual std::size_t test_size() const { return test_size_; }
    virtual std::size_t valid_size() const { return valid_size_; }
};

#endif

