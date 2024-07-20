//
// Created by peterglenn on 6/10/24.
//

#ifndef VERBOSEIO_H
#define VERBOSEIO_H
#include <iostream>
#include <string>

/*
#include "mysql_connection.h"
#include "cppconn/driver.h"
#include "cppconn/exception.h"
#include "cppconn/resultset.h"
#include "cppconn/statement.h"
*/
//#include <mysqlx/xdevapi.h>
// the above line is commented out because it won't compile; note the xdevapi route is more modern, I think

#define VERBOSE

/*
class verboseio {
public:
    verboseio() {};
    virtual void output( std::string str ) {
    }
};


class verbosedbio : public verboseio {
public:
    sql::Driver *driver;
    sql::Connection *con;

    verbosedbio( std::string hostName, std::string userName, std::string password, std::string catalog) {
        try {
            // Create a connection
            //driver = get_driver_instance();
            // the above line is commented out because it won't compile; and of course it is a needed line
            // for this code to run, so for now not utilizing the code here
            con = driver->connect(hostName,userName,password);
            // Connect to the MySQL test database
            con->setSchema(catalog);

        } catch (sql::SQLException &e) {
            std::cout << "# ERR: SQLException in " << __FILE__;
            std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
            std::cout << "# ERR: " << e.what();
            std::cout << " (MySQL error code: " << e.getErrorCode();
            std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
        }
    }

    ~verbosedbio() {
        delete con;
    }

    void output(std::string str);
};

class verbosestreamio : public verboseio {
public:
    std::ostream os;
    void output(std::string str);
};

class verbositylevels {

#ifdef VERBOSE
    bool verbose = true;
#else
    bool verbose = false;
#endif

};

inline void verbosedbio::output(std::string str) {
    try {
        sql::Statement *stmt;
        stmt = con->createStatement();
        stmt->execute("INSERT INTO LOG(txt) VALUES (" + str + ")");
        delete stmt;
    } catch (sql::SQLException &e) {
        std::cout << "# ERR: SQLException in " << __FILE__;
        std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
        std::cout << "# ERR: " << e.what();
        std::cout << " (MySQL error code: " << e.getErrorCode();
        std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
    }


}


inline void verbosestreamio::output(std::string str) {
    os << str;
}

*/
#endif //VERBOSEIO_H
