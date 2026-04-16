"""
슬리피지, 수수료, 포지션 사이징 모델.

각 모델은 베이스 클래스를 상속하여 구현.
새 모델이 필요하면 베이스 클래스를 상속해서 추가하면 된다.
"""
from __future__ import annotations


# =============================
# 슬리피지 모델
# =============================

class SlippageModel:
    """슬리피지(체결가 미끄러짐) 모델 인터페이스.
    실제 체결가 = 가정된 체결가(open 등)에 슬리피지 조정을 적용한 결과.
    """
    def slip_price(self, side: str, price: float) -> float:
        raise NotImplementedError


class BasicSlippage(SlippageModel):
    """고정 bps(베이시스 포인트) 슬리피지 모델.
    - 매수: 가격을 +bps만큼 불리하게
    - 매도: 가격을 -bps만큼 불리하게
    """
    def __init__(self, bps: float = 5.0):
        self.bps = bps

    def slip_price(self, side: str, price: float) -> float:
        adj = price * (self.bps / 10_000)
        return price + adj if side == "buy" else price - adj


# =============================
# 수수료 모델
# =============================

class CommissionModel:
    """수수료 모델 인터페이스. 공통 입력은 체결 금액(notional)."""
    def cost(self, notional: float) -> float:
        raise NotImplementedError


class FixedRateCommission(CommissionModel):
    """고정 비율 수수료 모델 (bps 단위).
    바이낸스 기본 수수료: maker/taker 각 0.1% = 10bps.
    """
    def __init__(self, bps: float = 10.0):
        self.bps = bps

    def cost(self, notional: float) -> float:
        return abs(notional) * (self.bps / 10_000)


# =============================
# 포지션 사이징 모델
# =============================

class Sizer:
    """포지션 사이징 인터페이스.
    signal(+1/-1)에 따라 얼마를 살지/팔지(수량) 결정.
    """
    def size(self, price: float, cash: float, pos_qty: float, signal: int) -> float:
        raise NotImplementedError


class FixedUSDTSizer(Sizer):
    """1회 매매 금액을 USDT 기준으로 고정하는 사이저.
    - 매수: USDT 금액/현재가 → 수량
    - 매도: 보유 수량 전량 매도 (단순화)
    """
    def __init__(self, usdt_per_trade: float = 100.0, min_qty: float = 0.0001):
        self.usdt = usdt_per_trade
        self.min_qty = min_qty

    def size(self, price: float, cash: float, pos_qty: float, signal: int) -> float:
        if signal > 0:  # buy
            qty = self.usdt / price
            return max(qty, self.min_qty)
        elif signal < 0:  # sell -> 전량 처분
            return -pos_qty
        return 0.0


class PercentSizer(Sizer):
    """자본 비례 사이저.
    현재 가용 현금의 일정 비율(percent)만큼을 1회 매매에 투입.
    - 매수: 현금 * percent / 현재가 -> 수량
    - 매도: 보유 수량 전량 매도
    """
    def __init__(self, percent: float = 0.15, min_qty: float = 0.0001):
        self.percent = percent   # 0.15 = 15%
        self.min_qty = min_qty

    def size(self, price: float, cash: float, pos_qty: float, signal: int) -> float:
        if signal > 0:  # buy
            usdt_amount = cash * self.percent
            qty = usdt_amount / price
            return max(qty, self.min_qty)
        elif signal < 0:  # sell -> 전량 처분
            return -pos_qty
        return 0.0


class FuturesSizer(Sizer):
    """선물 전용 사이저 — 롱/숏 양방향 지원.

    총 자산(equity)의 일정 비율로 목표 포지션을 설정하고,
    현재 포지션과의 차이(delta)만큼 주문을 낸다.

    signal=+1 : 롱 목표  (현재 숏이면 청산 후 롱 진입)
    signal=-1 : 숏 목표  (현재 롱이면 청산 후 숏 진입)
    signal= 0 : 플랫 목표 (현재 포지션 전량 청산)

    Returns
    -------
    delta : float
        양수 → buy  /  음수 → sell
    """
    def __init__(self, percent: float = 0.20, min_qty: float = 0.001):
        self.percent = percent
        self.min_qty = min_qty

    def size(self, price: float, cash: float, pos_qty: float, signal: int) -> float:
        if price <= 0:
            return 0.0

        equity = cash + pos_qty * price
        target_notional = equity * self.percent
        target_qty = target_notional / price

        if signal > 0:
            # 이미 롱이면 유지 (피라미딩 방지)
            if pos_qty > self.min_qty:
                return 0.0
            target = target_qty
        elif signal < 0:
            # 이미 숏이면 유지 (피라미딩 방지)
            if pos_qty < -self.min_qty:
                return 0.0
            target = -target_qty
        else:
            # signal=0 → 청산
            target = 0.0

        delta = target - pos_qty

        if abs(delta) < self.min_qty:
            return 0.0
        return delta
