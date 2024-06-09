//
// Created by peterglenn on 3/13/24.
//

#ifndef BATCHPROCESSEDDATA_H
#define BATCHPROCESSEDDATA_H


#include <utility>
#include <stdexcept>
#include <iostream>
#include <vector>

class Batchprocessed {
public:
    bool paused() const {return paused_;};
    bool changed() const {return changed_;}
    void setchanged(const bool b) {
        if (b && !paused())
            process();
        changed_ = b;
    }
    void pause() {
        paused_ = true;
    }
    void resume() {
        if (paused() && changed())
            process();
        paused_ = false;
    }
    virtual void process() {
        changed_ = false;
    }

    Batchprocessed() {}
private:
    bool paused_ {true};
    bool changed_ {true};

};

template<typename T>
class Batchprocesseddata : public Batchprocessed {
public:
    void readvector(const std::vector<T> datain) {
        auto p = paused();
        pause();
        if (data)
            delete[] data;
        size_ = datain.size();
        data = new T[size_]{};
        for (int n = 0; n < size_; ++n) {
            setdata(datain[n], n);
        }
        if (!p)
            resume();
    }
    Batchprocesseddata(const int s);
    Batchprocesseddata(std::vector<T> datain) : Batchprocessed() {
        data = nullptr;
        readvector(datain);
    }

    Batchprocesseddata();

    ~Batchprocesseddata();

    void setdata( T elem, const int i );
    virtual void setsize( int s );
    T getdata( const int i ) const;
    int size() const {return size_;};

    T operator[](int i) const {return getdata(i);}

    virtual void process() override {
        bool p = paused();
        pause();
        sortdata();
        removeduplicates();     // buggy because of initializing with all elements zero; it would delete everything if so
        Batchprocessed::process();
        if (!p)
            resume();
    }

    virtual void sortdata();
    virtual void removeduplicates();
private:
    T* data {nullptr};
    int size_ {0};
};



template<typename T>
inline Batchprocesseddata<T>::Batchprocesseddata(const int s) : Batchprocessed()
{
    if (s < 0)
        throw std::out_of_range("Index less than zero");
    data = new T[s];
    //for (int n = 0; n < s; ++n) {
    //    setdata(*(new T),n);
    //}
    size_ = s;
}

template<typename T>
inline Batchprocesseddata<T>::Batchprocesseddata() : Batchprocessed()
{
    data = new T[0];
    size_ = 0;
}

template<typename T>
inline Batchprocesseddata<T>::~Batchprocesseddata()
{
    //delete[] data;
}


template<typename T>
inline T Batchprocesseddata<T>::getdata(int i) const
{
    if (i <0 || size() <= i)
        throw std::out_of_range("getdata index outside of range");
    return data[i];
}


template<typename T>
inline void Batchprocesseddata<T>::setdata(T elem, int i)
{
    if (i <0 || size() <= i)
        throw std::out_of_range("setdata index outside of range" );
    data[i] = elem;
    setchanged(true);
}

template<typename T>
inline void Batchprocesseddata<T>::setsize(int s)
{
    if (data) {
        delete[] data;
    }
    data = new T[s];
    size_ = s;
    setchanged(true);
}

template<typename T>
inline void Batchprocesseddata<T>::sortdata() {
    bool ch = true;
    bool p = paused();
    int sz = size();
    pause();
    while (ch) {
        ch = false;
        for (int i = 0; i < (sz - 1); ++i) {
            if (getdata(i + 1) < getdata(i)) {
                T temp = getdata(i);
                setdata(getdata(i + 1), i);
                setdata(temp, i + 1);
                ch = true;
            }
        }
    }
    if (!p)
        resume();
}

template<typename T>
inline void Batchprocesseddata<T>::removeduplicates() {
    const bool p = paused();
    int newsize = size_;
    pause();
    int m = 0;
    while (m < newsize) {
        T d = getdata(m);
        bool found = false;
        int n = m+1;
        while (!found && n < newsize) {
            found = (getdata(n)==d);
            if (found) {
                for (int k = n; k < newsize-1; ++k)
                    setdata(getdata(k + 1), k);
                --newsize;
            } else
                ++n;
        }
        if (!found)
            ++m;
    }
    size_ = newsize;
    if (!p)
        resume();
}



#endif //BATCHPROCESSEDDATA_H
