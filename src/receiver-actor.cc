#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_tcp.h>

#include "lng/receiver-actor.h"

#include "log.h"

namespace lng {

void Payload::Clear()
{
    if (this->buf) {
        rte_pktmbuf_free(this->buf);
    }
    this->buf = nullptr;
    this->segments_num = 0;
}

uint32_t Payload::ExtractPayload(rte_mbuf* mbuf)
{
    this->buf = mbuf;

    auto seg = mbuf;
    int header_size = sizeof(rte_ether_hdr) + sizeof(rte_ipv4_hdr) + sizeof(rte_tcp_hdr);

    while (seg) {
        if (this->segments_num >= segments_max) {
            throw std::runtime_error("# of payload overflow");
        }

        uint16_t seg_len = rte_pktmbuf_data_len(seg) - header_size;
        if (seg_len > 0) {
            this->segments[this->segments_num].addr = rte_pktmbuf_mtod_offset(seg, uint8_t*, header_size);
            this->segments[this->segments_num].size = rte_pktmbuf_data_len(seg) - header_size;
            this->segments_num++;
        }
        seg = seg->next;

        header_size = 0;
    }
    auto* ipv4 = rte_pktmbuf_mtod_offset(mbuf, rte_ipv4_hdr*, sizeof(rte_ether_hdr));
    uint32_t payload_size = rte_be_to_cpu_16(ipv4->total_length) - sizeof(rte_ipv4_hdr) - sizeof(rte_tcp_hdr);

    return payload_size;
}

void Receiver::setup()
{
    log::debug("Receiver is waiting for 3-way handshake");
    nic_stream_->wait_for_3wayhandshake();
    log::debug("Receiver is awaked from 3-way handshake");
}

void Receiver::main()
{
    if (!payload_) {
        if (!ready_payload_stream_->get(&payload_, 1)) {
            return;
        }
        payload_->Clear();
    }

    rte_mbuf* v;
    if (!nic_stream_->get(&v, 1)) {
        return;
    }

    if (!nic_stream_->check_target_packet(v)) {
        return;
    }

    // TODO detect FIN and quit
    auto len = payload_->ExtractPayload(v);

    nic_stream_->send_ack(v, len);

    valid_payload_stream_->put(&payload_, 1);

    payload_ = nullptr;
}

#ifdef __AVX512F__

#pragma message("AVX512 selected")

void lng_memcpy(uint8_t* dst, uint8_t* src, size_t size)
{
    const uint8_t align = 64;
    uintptr_t dst_aligned = (((uintptr_t)(dst) + align - 1) / align) * align;
    if ((uintptr_t)dst != dst_aligned) {
        size_t s = dst_aligned - (uintptr_t)dst;
        memcpy((char*)dst, (char*)src, s);
        size -= s;
        src += s;
    }
    const uint8_t avx_len = 64;
    uintptr_t size_aligned = (size / avx_len) * avx_len;
    uintptr_t dst_p = dst_aligned;
    for (; dst_p < dst_aligned + size_aligned; dst_p += avx_len, src += avx_len) {
        _mm512_stream_si512((__m512i*)dst_p, _mm512_loadu_si512((__m512i*)src));
    }
    if (size_aligned != size) {
        memcpy((char*)dst_p, (char*)src, size - size_aligned);
    }
}

#elif __AVX2__
#pragma message("AVX2 selected")

void lng_memcpy(uint8_t* dst, uint8_t* src, size_t size)
{
    const uint8_t align = 64;
    uintptr_t dst_aligned = (((uintptr_t)(dst) + align - 1) / align) * align;
    if ((uintptr_t)dst != dst_aligned) {
        size_t s = dst_aligned - (uintptr_t)dst;
        memcpy((char*)dst, (char*)src, s);
        size -= s;
        src += s;
    }
    const uint8_t avx_len = 32;
    uintptr_t size_aligned = (size / avx_len) * avx_len;
    uintptr_t dst_p = dst_aligned;
    for (; dst_p < dst_aligned + size_aligned; dst_p += avx_len, src += avx_len) {
        _mm256_stream_si256((__m256i*)dst_p, _mm256_loadu_si256((__m256i*)src));
    }
    if (size_aligned != size) {
        memcpy((char*)dst_p, (char*)src, size - size_aligned);
    }
}
#else
#warning "nomal memcpy"
void lng_memcpy(uint8_t* dst, uint8_t* src, size_t size)
{
    memcpy(dst, src, size);
}
#endif

void FrameBuilder::main()
{
    if (!frame_) {
        if (!ready_frame_stream_->get(&frame_, 1)) {
            return;
        }
    }

    if (!payload_) {
        if (!valid_payload_stream_->get(&payload_, 1)) {
            return;
        }
    }

    int seg;
    for (seg = payload_segment_id_; seg < payload_->segments_num; seg++) {
        auto segment_size = payload_->segments[seg].size - payload_segment_read_offset_;
        if (frame_write_offset_ + segment_size > Frame::frame_size) {
            // The payload has to be split into two frames
            size_t copy_size = Frame::frame_size - frame_write_offset_;
            log::trace("1:w({:5}) <- r({:5},{:5}) len({:5}) val({:#x})", frame_write_offset_, payload_segment_id_, payload_segment_read_offset_, copy_size, *(payload_->segments[seg].addr + payload_segment_read_offset_));
            lng_memcpy(frame_->body + frame_write_offset_, payload_->segments[seg].addr + payload_segment_read_offset_, copy_size);
            
            // Store payload segment id and read offset for the next frame
            payload_segment_id_ = seg;
            payload_segment_read_offset_ = segment_size - copy_size;
            
            // Frame is pushed to the stream
            frame_->frame_id = this->frame_id_++;
            valid_frame_stream_->put(&frame_, 1);
            frame_ = nullptr;
            frame_write_offset_ = 0;
            break;
        } else if (frame_write_offset_ + segment_size == Frame::frame_size) {
            // The payload can fit into the frame exactly
            log::trace("2:w({:5}) <- r({:5},{:5}) len({:5}) val({:#x})", frame_write_offset_, payload_segment_id_, payload_segment_read_offset_, segment_size, *(payload_->segments[seg].addr + payload_segment_read_offset_));
            lng_memcpy(frame_->body + frame_write_offset_, payload_->segments[seg].addr + payload_segment_read_offset_, segment_size);
            
            // Payload segment id and read offset should be point to the next segment because the payload is fully copied
            payload_segment_id_ = seg + 1;
            payload_segment_read_offset_ = 0;
            
            // Frame is pushed to the stream
            frame_->frame_id = this->frame_id_++;
            valid_frame_stream_->put(&frame_, 1);
            frame_ = nullptr;
            frame_write_offset_ = 0;
            break;
        } else {
            // The payload can fit into the frame with some space left
            log::trace("3:w({:5}) <- r({:5},{:5}) len({:5}) val({:#x})", frame_write_offset_, payload_segment_id_, payload_segment_read_offset_, segment_size, *(payload_->segments[seg].addr + payload_segment_read_offset_));
            lng_memcpy(frame_->body + frame_write_offset_, payload_->segments[seg].addr + payload_segment_read_offset_, segment_size);
            payload_segment_read_offset_ = 0;
            frame_write_offset_ += segment_size;
        }
    }

    // Release payload if it is fully copied
    if (seg == payload_->segments_num && payload_segment_read_offset_ == 0) {
        ready_payload_stream_->put(&payload_, 1);
        payload_ = nullptr;
        payload_segment_id_ = 0;
        payload_segment_read_offset_ = 0;
    }
}

} // lng
