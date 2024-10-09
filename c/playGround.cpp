#include "util.h"

class Serializable
{
};

class Dataclass : public Serializable
{
public:
    int a;
    int b;
};

int main()
{
    return 0;
}
