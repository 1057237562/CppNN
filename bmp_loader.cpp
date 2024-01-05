#ifndef BMP_LOADER_CPP
#define BMP_LOADER_CPP

#include "mutil.cpp"
#include <fstream>
#include <iostream>

struct BitmapFileHeader {
    char type[2];
    unsigned int size;
    short reserved1;
    short reserved2;
    unsigned int offset;

    friend ostream& operator<<(ostream& os, const BitmapFileHeader& header)
    {
        os << header.type[0] << header.type[1];
        os << header.size;
        os << header.reserved1;
        os << header.reserved2;
        os << header.offset;
        return os;
    }

    friend istream& operator>>(istream& is, BitmapFileHeader& header)
    {
        is >> header.type[0] >> header.type[1];
        is.read((char*)&header.size, sizeof(header.size));
        is.read((char*)&header.reserved1, sizeof(header.reserved1));
        is.read((char*)&header.reserved2, sizeof(header.reserved2));
        is.read((char*)&header.offset, sizeof(header.offset));
        return is;
    }
};

struct BitmapInfoHeader {
    unsigned int size;
    unsigned int width;
    unsigned int height;
    unsigned short planes;
    unsigned short bitCount;
    unsigned int compression;
    unsigned int sizeImage;
    unsigned int xPelsPerMeter;
    unsigned int yPelsPerMeter;
    unsigned int clrUsed;
    unsigned int clrImportant;

    friend ostream& operator<<(ostream& os, const BitmapInfoHeader& header)
    {
        os << header.size;
        os << header.width;
        os << header.height;
        os << header.planes;
        os << header.bitCount;
        os << header.compression;
        os << header.sizeImage;
        os << header.xPelsPerMeter;
        os << header.yPelsPerMeter;
        os << header.clrUsed;
        os << header.clrImportant;
        return os;
    }

    friend istream& operator>>(istream& is, BitmapInfoHeader& header)
    {
        is.read((char*)&header.size, sizeof(header.size));
        is.read((char*)&header.width, sizeof(header.width));
        is.read((char*)&header.height, sizeof(header.height));
        is.read((char*)&header.planes, sizeof(header.planes));
        is.read((char*)&header.bitCount, sizeof(header.bitCount));
        is.read((char*)&header.compression, sizeof(header.compression));
        is.read((char*)&header.sizeImage, sizeof(header.sizeImage));
        is.read((char*)&header.xPelsPerMeter, sizeof(header.xPelsPerMeter));
        is.read((char*)&header.yPelsPerMeter, sizeof(header.yPelsPerMeter));
        is.read((char*)&header.clrUsed, sizeof(header.clrUsed));
        is.read((char*)&header.clrImportant, sizeof(header.clrImportant));
        return is;
    }
};

struct Color {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
    unsigned char reserved;

    friend ostream& operator<<(ostream& os, const Color& color)
    {
        os << color.blue;
        os << color.green;
        os << color.red;
        os << color.reserved;
        return os;
    }

    friend istream& operator>>(istream& is, Color& color)
    {
        is.read((char*)&color.blue, sizeof(color.blue));
        is.read((char*)&color.green, sizeof(color.green));
        is.read((char*)&color.red, sizeof(color.red));
        is.read((char*)&color.reserved, sizeof(color.reserved));
        return is;
    }
};

mutil::Mat readBmp(string filename)
{
    mutil::Mat img;
    ifstream f(filename, ios::binary);
    if (!f.is_open()) {
        std::cout << "Error: File not found" << std::endl;
        return img;
    }
    BitmapFileHeader fileHeader;
    BitmapInfoHeader infoHeader;
    f >> fileHeader;
    f >> infoHeader;
    if (infoHeader.bitCount > 16) {
        f.seekg(54, ios::beg);
        infoHeader.clrUsed = infoHeader.clrImportant = 0;
    } else if (!infoHeader.clrUsed) {
        infoHeader.clrUsed = infoHeader.bitCount;
    }
    Color color[infoHeader.clrUsed];
    for (Color& c : color) {
        f >> c;
    }
    int byteCount = infoHeader.bitCount >> 3;
    int size = byteCount * infoHeader.width * infoHeader.size;
    char data[size];
    f.read(data, sizeof(data));
    f.close();
    img = mutil::Mat(3, infoHeader.width * infoHeader.size);
    for (int i = 0; i < infoHeader.height; i++) {
        int row = infoHeader.height - 1 - i;
        for (int j = 0; j < infoHeader.width; j++) {
            int col = j;
            if (infoHeader.bitCount > 16) {
                img[0][row * infoHeader.width + col] = data[byteCount * (i * infoHeader.width + j) + 2] / 255.0;
                img[1][row * infoHeader.width + col] = data[byteCount * (i * infoHeader.width + j) + 1] / 255.0;
                img[2][row * infoHeader.width + col] = data[byteCount * (i * infoHeader.width + j)] / 255.0;
            } else {
                int index = data[byteCount * (i * infoHeader.width + j)];
                img[0][row * infoHeader.width + col] = color[index].red / 255.0;
                img[1][row * infoHeader.width + col] = color[index].green / 255.0;
                img[2][row * infoHeader.width + col] = color[index].blue / 255.0;
            }
        }
    }
    return img;
}

#endif