
#ifndef iFile_hpp
#define iFile_hpp

	#include "CharArray.hpp"
	#include "ArrayPointer.hpp"
	#include "MemorySize.hpp"

	#if defined(pankey_Log) && (defined(iFile_Log) || defined(pankey_Global_Log) || defined(pankey_FileSystem_Log))
		#include "Logger_status.hpp"
		#define iFileLog(status,method,mns) pankey_Log(status,"iFile",method,mns)
	#else
		#define iFileLog(status,method,mns)
	#endif


	namespace pankey{

		namespace FileSystem{

			class iFile{
				public:
					virtual ~iFile(){}

					virtual void initialize()=0;

					virtual void createFolder(const Base::CharArray& a_filename)=0;
					virtual void deleteFolder(const Base::CharArray& a_filename)=0;
					virtual bool folderExist(const Base::CharArray& a_filename)=0;
					
					virtual bool open(const Base::CharArray& a_filename, const Base::CharArray& a_mode)=0;
					virtual void close()=0;
					virtual int read(Base::CharArray& a_buffer, int a_count)=0;
					virtual Base::CharArray read(int a_count)=0;
					virtual int read(void* a_data, int a_count, Base::memory_size a_size)=0;
					virtual void write(const Base::CharArray& a_data)=0;
					virtual void write(const void* a_data, int a_count, Base::memory_size a_size)=0;
					virtual bool seek(int a_position)=0;
					virtual bool isOpen()const=0;

					virtual Base::CharArray pathSeparator()const=0;

					template<class T>
					void readArray(Base::ArrayPointer<T>& a_values, int a_count){
						iFileLog(pankey_Log_StartMethod, "readArray",  "");
						if(a_values.getSize() < a_count){
							a_values.createArray(a_count);
						}
						a_values.reset();
						void* i_pointer = (void*)a_values.getArrayPointer();
						Base::memory_size i_size = sizeof(T);
						int i_read = this->read(i_pointer, a_count, i_size);
						int i_length = i_read / i_size;
						a_values.setLength(i_length);
			
						iFileLog(pankey_Log_Statement, "readArray",  "Total bytes read");
						iFileLog(pankey_Log_Statement, "readArray",  i_length);
			
						iFileLog(pankey_Log_EndMethod, "readArray",  "");
					}

					template<class T>
					void writeArray(const Base::ArrayPointer<T>& a_values, int a_count){
						if(a_values.isEmpty()){
							return;
						}
						void* i_pointer = (void*)a_values.getArrayPointer();
						Base::memory_size i_size = sizeof(T);
						this->write(i_pointer, a_count, i_size);
					}
			};
		}

	}

#endif