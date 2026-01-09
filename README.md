# Traffic Light

**Author:** Landon Quaintance (`lquainta`)

## About

Traffic Light detects traffic lights and plays a chime when a red light switches to green after being red for at least 5 seconds. This delay helps ignore transient light changes that might occur while driving.

## Process

This project was a small learning experiment with Machine Learning and AI Delegation. The development workflow involved two AI chatbots with distinct roles:

1. **Code Reviewer:** Analyzed the original code and provided feedback.
2. **Code Writer:** Implemented changes based on the reviewer’s feedback.

The process was iterative: the writer’s updated code was sent back to the reviewer, and the reviewer’s feedback was sent back to the writer. This loop continued until the code reached a production-ready state.

## Installation

This project requires Python 3 and the following packages:

```bash
pip install numpy opencv-python
```

## Usage

1. Connect a camera to your system.
2. Run the script:

```bash
python traffic_light.py
```

3. The program will detect traffic lights and play a chime when a red light turns green after at least 5 seconds.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
