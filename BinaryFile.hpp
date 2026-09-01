#pragma once

#include "CharArray.hpp"

namespace pankey{

    namespace FileSystem{

        using FilePath = pankey::Type::Array::CharArray;

        class BinaryFile{
            public:
                virtual ~BinaryFile(){}

                virtual bool openRead(const FilePath& a_path) = 0;
                virtual bool openWrite(const FilePath& a_path) = 0;
                virtual void close() = 0;
                virtual bool isOpen() const = 0;
                virtual bool skip(unsigned long long a_byte_count) = 0;
                virtual bool seek(unsigned long long a_position) = 0;
                virtual unsigned long long position() const = 0;

                template<class Value>
                bool read(Value& a_value){
                    return readBytes(&a_value, sizeof(Value));
                }

                template<class Value>
                bool write(const Value& a_value){
                    return writeBytes(&a_value, sizeof(Value));
                }

                template<class Policy>
                bool readArray(
                    pankey::DataStructure::Array::ArrayPointer<Policy>& a_values,
                    int a_count
                ){
                    if(a_count < 0 || a_values.getSize() < a_count){
                        return false;
                    }
                    if(a_count == 0){
                        a_values.setLength(0);
                        return true;
                    }
                    if(!readBytes(
                        a_values.getArrayPointer(),
                        static_cast<int>(sizeof(typename Policy::VALUE_TYPE) * a_count)
                    )){
                        return false;
                    }
                    a_values.setLength(a_count);
                    return true;
                }

            protected:
                virtual bool readBytes(void* a_values, int a_byte_count) = 0;
                virtual bool writeBytes(const void* a_values, int a_byte_count) = 0;
        };
    }
}