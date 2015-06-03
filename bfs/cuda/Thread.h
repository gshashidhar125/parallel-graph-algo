using namespace std;
class ThreadClass
{
public:
   ThreadClass() {}

   virtual ~ThreadClass() {}

   /** Returns true if the thread was successfully started, false if there was an error starting the thread */
   bool StartInternalThread()
   {
      return (pthread_create(&_thread, NULL, InternalThreadEntryFunc, this) == 0);
   }

   /** Will not return until the internal thread has exited. */
   void WaitForInternalThreadToExit()
   {
      (void) pthread_join(_thread, NULL);
   }
   void setThreadId(int a) {
        threadId = a;
   }
   int getThreadId() {
        return threadId;
   }

protected:
   /** Implement this method in your subclass with the code you want your thread to run. */
   virtual void InternalThreadEntry() = 0;

private:
   static void * InternalThreadEntryFunc(void * This) {((ThreadClass *)This)->InternalThreadEntry(); return NULL;}

   pthread_t _thread;
   int threadId;
};


class parallelBFS : public ThreadClass {

public:
//private:
    GraphClass *graphData;
    parallelBFS() : ThreadClass() {}
    parallelBFS(int a) : ThreadClass() {setThreadId(a);}
    void setGraph(GraphClass *a) {
        graphData = a;
    }
    void InternalThreadEntry();
};
