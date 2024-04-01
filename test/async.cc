#include <iostream>
#include <cassert>
#include <coroutine>

class resumable {
public:
	struct promise_type;
	using coro_handle = std::coroutine_handle<promise_type>;

	resumable(coro_handle &handle) : co_handle(handle) { assert(handle); }
	resumable(resumable &) = delete;
	resumable(resumable &&) = delete;

	~resumable() {
		co_handle.destroy();
	}

	bool resume() {
		if (!co_handle.done()) {
			co_handle.resume();
		}
		return !co_handle.done();
	}
private:
	coro_handle co_handle;
};

struct resumable::promise_type {
	using coro_handle = std::coroutine_handle<promise_type>;

	auto get_return_object() {
		return coro_handle::from_promise(*this);
	}

	auto initial_suspend() { return std::suspend_always(); }
	auto final_suspend() noexcept { return std::suspend_always(); }
	void return_void() {}
	void unhandled_exception() {
		std::terminate();
	}
};

resumable foo() {
	std::cout << "First" << std::endl;
	co_await std::suspend_always();
	std::cout << "Second" << std::endl;
}

int main() {
	auto p = foo();
	while (p.resume());
	return 0;
}
