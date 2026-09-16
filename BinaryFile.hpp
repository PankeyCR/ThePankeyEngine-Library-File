#pragma once

#include "IFile.hpp"
#include "FileManager.hpp"

namespace pankey{

    namespace FileSystem{

        using FilePath = pankey::Support::File::FilePath;
        using FileMode = pankey::Support::File::FileMode;
        using ReadStatus = pankey::Support::File::ReadStatus;
        using IFile = pankey::Support::File::IFile;
        using FileManager = pankey::Support::File::FileManager;

        // Thin adapter kept for the existing GGUF / NeuralNetwork readers.
        // New code should use Support::File::FileManager directly.
        class BinaryFile{
            public:
                BinaryFile(IFile& a_file) : m_files(a_file){}

                bool openRead(const FilePath& a_path){
                    return m_files.openRead(a_path);
                }

                bool openWrite(const FilePath& a_path){
                    return m_files.openWrite(a_path);
                }

                void close(){
                    m_files.close();
                }

                bool isOpen() const{
                    return m_files.isOpen();
                }

                bool skip(unsigned long long a_byte_count){
                    return m_files.skip(a_byte_count);
                }

                bool seek(unsigned long long a_position){
                    return m_files.seek(a_position);
                }

                unsigned long long position() const{
                    return m_files.position();
                }

                template<class Value>
                bool read(Value& a_value){
                    return m_files.read(a_value);
                }

                template<class Value>
                bool write(const Value& a_value){
                    return m_files.write(a_value);
                }

                template<class Policy>
                bool readArray(
                    pankey::DataStructure::Array::ArrayPointer<Policy>& a_values,
                    int a_count
                ){
                    return m_files.readArray(a_values, a_count);
                }

            protected:
                FileManager m_files;
        };
    }
}