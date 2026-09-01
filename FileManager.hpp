
#ifndef FileManager_hpp
#define FileManager_hpp

	#include "iFile.hpp"
	#include "CharArray.hpp"

	#if defined(pankey_Log) && (defined(FileManager_Log) || defined(pankey_Global_Log) || defined(pankey_Base_Log))
		#include "Logger_status.hpp"
		#define FileManagerLog(status,method,mns) pankey_Log(status,"FileManager",method,mns)
	#else
		#define FileManagerLog(status,method,mns)
	#endif


	namespace pankey{

		namespace FileSystem{

			class FileManager{
				public:
					FileManager(iFile& a_file) : m_file(a_file){}
					virtual ~FileManager(){}

					virtual bool createDir(const CharArray& a_path){
						// ArduinoSDFileLog(pankey_Log_StartMethod, "createDir",  "println", "");
						// ArduinoSDFileLog(pankey_Log_Statement, "createDir",  "println", "Path: ");
						// ArduinoSDFileLog(pankey_Log_Statement, "createDir",  "println", a_path);
						// if(m_sd.mkdir(a_path.toString())){
						// 	ArduinoSDFileLog(pankey_Log_Statement, "createDir",  "println", "Dir created");
						// 	ArduinoSDFileLog(pankey_Log_EndMethod, "createDir",  "println", "");
						// 	return true;
						// }else{
						// 	ArduinoSDFileLog(pankey_Log_Statement, "createDir",  "println", "mkdir failed");
						// }
						// ArduinoSDFileLog(pankey_Log_EndMethod, "createDir",  "println", "");
						return false;
					}

					virtual bool dirExists(const CharArray& a_path){
						return false;
					}

					virtual bool deleteDir(const CharArray& a_path){
						return false;
					}
					virtual bool createFile(const CharArray& a_path){
						return false;
					}
					virtual bool deleteFile(const CharArray& a_path){
						return false;
					}
					virtual bool clearFile(const CharArray& a_path){
						return false;
					}
					virtual bool writeText(const CharArray& a_text, const CharArray& a_path){
						return false;
					}

				protected:
					iFile& m_file;
			};
		}

	}

#endif