#include "lng/receiver-actor.h"

#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_tcp.h>

#include <iostream>

namespace lng {

void Payloads::Clear()
{
    if (this->buf) {
        rte_pktmbuf_free(this->buf);
    }
    this->buf = nullptr;
    this->no_of_payload = 0;
}

size_t Payloads::ExtractPayloads(rte_mbuf* mbuf)
{
    size_t total_length = 0;

    this->buf = mbuf;

    auto seg = mbuf;
    int header_size = sizeof(rte_ether_hdr) + sizeof(rte_ipv4_hdr) + sizeof(rte_tcp_hdr);

    while (seg) {
        if (this->no_of_payload >= max_payloads) {
            throw std::runtime_error("# of payload overflow");
        }

        uint16_t seg_len = rte_pktmbuf_data_len(seg) - header_size;
        if (seg_len > 0) {
            this->segments[this->no_of_payload].payload = rte_pktmbuf_mtod_offset(seg, uint8_t*, header_size);
            this->segments[this->no_of_payload].length = rte_pktmbuf_data_len(seg) - header_size;
            this->no_of_payload++;
            total_length += seg_len;
        }
        seg = seg->next;

        header_size = 0;
    }

    return total_length;
}

void Receiver::main()
{
    Payloads* pays;
    if (!ready_payload_stream_->get(&pays)) {
        return;
    } else {
        pays->Clear();
    }

    while (true) {
        rte_mbuf* v;
        if (nic_stream_->get(&v)) {
            // std::cout << "received " << v->pkt_len << " bytes" << std::endl;

            auto len = pays->ExtractPayloads(v);

            nic_stream_->send_ack(v, len);

            vaild_payload_stream_->put(pays);
            break;
        }
    }
}

void FrameBuilder::main()
{
    if (!next_frame_) {
        if (!ready_frame_stream_->get(&next_frame_)) {
            return;
        }
    }

    Frame* frame = next_frame_;

    if (!ready_frame_stream_->get(&next_frame_)) {
        return;
    }

    bool complete = false;

    while (!complete) {
        Payloads* pays;
        if (vaild_payload_stream_->get(&pays)) {
            for (int seg = 0; seg < pays->no_of_payload; seg++) {
                auto len = pays->segments[seg].length;
                if (write_head_ + len < Frame::frame_size) {
                    memcpy(frame->body + write_head_, pays->segments[seg].payload, len);
                    write_head_ += len;
                } else if (write_head_ < Frame::frame_size) {
                    size_t bytes_cur_frame = write_head_ + len - Frame::frame_size;
                    size_t bytes_next_frame = len - bytes_cur_frame;
                    uint8_t* p = pays->segments[seg].payload;
                    memcpy(frame->body + write_head_, p, bytes_cur_frame);
                    memcpy(next_frame_->body, p + bytes_cur_frame, bytes_next_frame);
                    write_head_ = bytes_next_frame;

                    complete = true;
                } else {
                    memcpy(next_frame_->body + write_head_, pays->segments[seg].payload, len);
                    write_head_ += len;
                }
            }
            ready_payload_stream_->put(pays);
        }
    }
    frame->frame_id = this->frame_id_++;
    vaild_frame_stream_->put(frame);
}
}
