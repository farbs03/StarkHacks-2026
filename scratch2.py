from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader

leader = SO101Leader(
    SO101LeaderConfig(
        port="/dev/tty.usbmodem5AE60536821",
        id="leader_arm",
    )
)

leader.connect()